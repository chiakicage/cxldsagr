import hashlib
import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from experiments.gr_serving.src.workload import WorkloadConfig, build_workload, main, token_sha256
from GR.heat import HeatPopulation
from GR.input_generator import MODEL_FORMATS, InputGenerator, TextConfig, create_input_generator


@pytest.fixture
def tokenizer():
    # A real byte tokenizer makes tests independent of local checkpoints, while
    # still exercising GR's exact text encoding and model-specific boundaries.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special = sorted({token for fmt in MODEL_FORMATS.values() for token in fmt.SPECIAL_TOKENS})
    tokenizer.train_from_iterator(
        ["Readable text for tokenizer construction."],
        trainers.BpeTrainer(
            vocab_size=256 + len(special),
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tokenizer


@pytest.mark.parametrize("model", ["deepseek_v32", "nosa"])
def test_reproducible_weighted_requests_and_exact_prefix_identity(tokenizer, model):
    config = WorkloadConfig(
        model=model, num_users=3, requests=16, history_tokens=1024, candidate_tokens=1024
    )
    first = build_workload(config, tokenizer=tokenizer)
    second = build_workload(config, tokenizer=tokenizer)
    assert first == second
    assert first.manifest["schedule"]["sampling"] == "weighted"
    assert first.manifest["heat"]["dataset"] == "beauty"
    assert first.manifest["observed"]["revisits"] > 0
    previous = {}
    visits = {}
    for request in first.requests:
        uid = request["user_id"]
        assert request["stable_prefix_tokens"] == 1024
        assert request["candidate_suffix_tokens"] == 1024
        assert len(request["input_ids"]) == 2048
        assert request["visit_index"] == visits.get(uid, 0)
        assert request["is_revisit"] == (uid in previous)
        assert token_sha256(request["input_ids"]) == request["input_sha256"]
        assert (
            request["input_ids"]
            == tokenizer.encode(request["prompt"], add_special_tokens=False).ids
        )
        if uid in previous:
            old = previous[uid]
            assert old["input_ids"][:1024] == request["input_ids"][:1024]
            assert old["prefix_sha256"] == request["prefix_sha256"]
            assert old["candidate_sha256"] != request["candidate_sha256"]
        previous[uid] = request
        visits[uid] = request["visit_number"]
    assert sum(row["visits"] for row in first.manifest["users"]) == 16


def test_seed_changes_trace_and_write_roundtrip(tokenizer, tmp_path):
    common = {
        "model": "nosa",
        "num_users": 3,
        "requests": 8,
        "history_tokens": 1024,
        "candidate_tokens": 1024,
    }
    first = build_workload(WorkloadConfig(**common, seed=42), tokenizer=tokenizer)
    second = build_workload(WorkloadConfig(**common, seed=43), tokenizer=tokenizer)
    assert first.manifest["workload_sha256"] != second.manifest["workload_sha256"]
    first.write(tmp_path)
    assert json.loads((tmp_path / "workload.json").read_text()) == first.manifest
    assert [
        json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()
    ] == list(first.requests)
    with pytest.raises(FileExistsError):
        second.write(tmp_path)


def test_iid_trace_reports_actual_users_without_forced_coverage(tokenizer):
    workload = build_workload(
        WorkloadConfig(
            model="nosa", num_users=8, requests=1, history_tokens=1024, candidate_tokens=1024
        ),
        tokenizer=tokenizer,
    )
    assert workload.manifest["config"]["max_revisits"] is None
    assert workload.manifest["config"]["sampling"] == "weighted"
    assert workload.manifest["observed"]["unique_users"] == 1
    assert workload.manifest["observed"]["returning_users"] == 0
    assert workload.manifest["observed"]["revisits"] == 0
    assert sum(user["visits"] for user in workload.manifest["users"]) == 1
    assert all(user["revisits"] == 0 for user in workload.manifest["users"])


def test_uncapped_single_user_can_revisit_more_than_eight_times(tokenizer):
    workload = build_workload(
        WorkloadConfig(
            model="nosa", num_users=1, requests=13, history_tokens=1024, candidate_tokens=1024
        ),
        tokenizer=tokenizer,
    )
    assert len(workload.requests) == 13
    assert workload.manifest["observed"]["returning_users"] == 1
    assert workload.manifest["users"][0]["revisits"] == 12


@pytest.mark.parametrize(
    "override",
    [
        {"num_users": 0},
        {"requests": 0},
        {"seed": -1},
        {"history_tokens": 32768},
        {"heat_dataset": ""},
        {"heat_dataset": None},
        {"heat_field": ""},
        {"heat_field": 1},
        {"sampling": "uniform"},
        {"max_revisits": -1},
        {"max_revisits": True},
        {"max_revisits": 2.5},
        {"context_limit": 0},
        {"context_limit": -1},
        {"context_limit": True},
        {"context_limit": 2.5},
        {"context_limit": 262145},
    ],
)
def test_invalid_configuration(override):
    arguments = {"model": "nosa", "num_users": 2, "requests": 4}
    arguments.update(override)
    with pytest.raises(ValueError):
        WorkloadConfig(**arguments)


def test_too_short_fixed_prefix_rejected(tokenizer):
    config = WorkloadConfig(model="nosa", num_users=1, requests=1, history_tokens=2)
    with pytest.raises(ValueError, match="fixed instruction"):
        build_workload(config, tokenizer=tokenizer)


@pytest.mark.parametrize("model", ["deepseek_v32", "nosa"])
@pytest.mark.parametrize("cap", [0, 8])
def test_exhausted_workload_reports_actual_length_and_visit_identities(tokenizer, model, cap):
    workload = build_workload(
        WorkloadConfig(
            model=model,
            num_users=1,
            requests=128,
            max_revisits=cap,
            history_tokens=1024,
            candidate_tokens=1024,
        ),
        tokenizer=tokenizer,
    )
    assert len(workload.requests) == cap + 1
    assert workload.manifest["config"]["requests"] == 128
    assert workload.manifest["config"]["max_revisits"] == cap
    assert workload.manifest["schedule"]["max_revisits"] == cap
    assert workload.manifest["observed"] == {
        "requests": cap + 1,
        "unique_users": 1,
        "returning_users": int(cap > 0),
        "first_visits": 1,
        "revisits": cap,
        "max_visits": cap + 1,
        "max_revisits": cap,
    }
    for visit, (request, identity) in enumerate(
        zip(workload.requests, workload.manifest["requests"], strict=True)
    ):
        assert request["request_id"] == request["visit_index"] == visit
        assert request["visit_number"] == visit + 1
        assert request["is_revisit"] == (visit > 0)
        assert request["input_sha256"] == identity["input_sha256"]
        assert token_sha256(request["input_ids"]) == identity["input_sha256"]
    assert len({row["prefix_sha256"] for row in workload.requests}) == 1
    assert len({row["candidate_sha256"] for row in workload.requests}) == cap + 1


def test_workload_cli_propagates_revisit_cap(tokenizer, tmp_path, capsys):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    output = tmp_path / "output"
    main(
        [
            "--model",
            "nosa",
            "--num-users",
            "1",
            "--requests",
            "128",
            "--max-revisits",
            "1",
            "--heat-dataset",
            "industrial_10M",
            "--heat-field",
            "pv_share",
            "--history-tokens",
            "1024",
            "--candidate-tokens",
            "1024",
            "--context-limit",
            "2048",
            "--tokenizer",
            str(tokenizer_path),
            "--output-dir",
            str(output),
        ]
    )
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["requests"] == 2
    assert emitted["max_revisits"] == 1
    manifest = json.loads((output / "workload.json").read_text())
    assert manifest["config"]["max_revisits"] == 1
    assert manifest["config"]["context_limit"] == 2048
    assert manifest["config"]["heat_dataset"] == "industrial_10M"
    assert manifest["config"]["heat_field"] == "pv_share"
    assert manifest["context"]["effective_limit_tokens"] == 2048
    assert manifest["context"]["override_explicit"] is True
    assert len((output / "requests.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("model", ["deepseek_v32", "nosa"])
def test_industrial_accesses_match_published_trace_prefix(tokenizer, model):
    # These are the first eight rows of the archived industrial 10M / pv_share
    # 1024-user, seed-42 trace. Model formatting cannot change access identities.
    workload = build_workload(
        WorkloadConfig(
            model=model,
            num_users=1024,
            requests=8,
            history_tokens=1024,
            candidate_tokens=1024,
            heat_dataset="industrial_10M",
        ),
        tokenizer=tokenizer,
    )
    assert workload.manifest["config"]["heat_field"] == "pv_share"
    assert workload.manifest["heat"]["dataset"] == "industrial_10M"
    assert workload.manifest["heat"]["field"] == "pv_share"
    assert workload.manifest["text_config"]["history_cache_users"] == 8
    assert [row["user_id"] for row in workload.requests] == [631, 13, 269, 219, 744, 671, 915, 68]
    access_trace = [
        {
            "request_id": i,
            "user_id": uid,
            "visit_index": 0,
            "previous_request_id": None,
            "timestamp": float(i),
        }
        for i, uid in enumerate([631, 13, 269, 219, 744, 671, 915, 68])
    ]
    expected = hashlib.sha256(
        json.dumps(access_trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert workload.manifest["access_trace_sha256"] == expected


def test_explicit_curve_field_is_validated_by_gr(tokenizer):
    config = WorkloadConfig(
        model="nosa",
        num_users=1,
        requests=1,
        history_tokens=1024,
        heat_dataset="industrial_10M",
        heat_field="interaction_count",
    )
    with pytest.raises(ValueError, match="no curve for industrial_10M/interaction_count"):
        build_workload(config, tokenizer=tokenizer)


@pytest.mark.parametrize("model", ["deepseek_v32", "nosa"])
def test_sequential_two_passes_have_no_heat_dependency(tokenizer, monkeypatch, model):
    def forbidden_curve(*args, **kwargs):
        raise AssertionError("sequential input must not read a heat curve")

    monkeypatch.setattr(HeatPopulation, "from_curve", forbidden_curve)
    workload = build_workload(
        WorkloadConfig(
            model=model,
            num_users=16,
            requests=32,
            history_tokens=1024,
            candidate_tokens=1024,
            sampling="sequential",
            heat_dataset="a-curve-that-does-not-exist",
            heat_field="not-a-heat-field",
        ),
        tokenizer=tokenizer,
    )
    manifest = workload.manifest
    assert manifest["config"]["heat_dataset"] is None
    assert manifest["config"]["heat_field"] is None
    assert manifest["heat_sha256"] is None
    assert manifest["schedule"]["sampling"] == "sequential"
    assert manifest["heat"]["source"] == "explicit_synthetic_ids"
    assert {"path", "sha256", "dataset", "field"}.isdisjoint(manifest["heat"])
    assert all(user["weight"] == 1.0 and user["probability"] is None for user in manifest["users"])
    assert manifest["observed"] == {
        "requests": 32,
        "unique_users": 16,
        "returning_users": 16,
        "first_visits": 16,
        "revisits": 16,
        "max_visits": 2,
        "max_revisits": 1,
    }
    access_trace = []
    for rid, request in enumerate(workload.requests):
        access = {
            "request_id": rid,
            "user_id": rid % 16,
            "visit_index": rid // 16,
            "previous_request_id": rid - 16 if rid >= 16 else None,
            "timestamp": float(rid),
        }
        assert {key: request[key] for key in access} == access
        assert request["is_revisit"] == (rid >= 16)
        if rid >= 16:
            previous = workload.requests[rid - 16]
            assert request["prefix_sha256"] == previous["prefix_sha256"]
            assert request["candidate_sha256"] != previous["candidate_sha256"]
        access_trace.append(access)
    assert (
        manifest["access_trace_sha256"]
        == hashlib.sha256(
            json.dumps(access_trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("cap", [0, 1, 8])
def test_sequential_rejects_revisit_cap(cap):
    with pytest.raises(ValueError, match="sequential sampling does not support max_revisits"):
        WorkloadConfig("nosa", 16, 32, sampling="sequential", max_revisits=cap)


def test_workload_cli_sequential_records_no_heat(tokenizer, tmp_path, capsys):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    output = tmp_path / "output"
    main(
        [
            "--model",
            "nosa",
            "--num-users",
            "2",
            "--requests",
            "4",
            "--sampling",
            "sequential",
            "--history-tokens",
            "1024",
            "--candidate-tokens",
            "1024",
            "--tokenizer",
            str(tokenizer_path),
            "--output-dir",
            str(output),
        ]
    )
    assert json.loads(capsys.readouterr().out)["requests"] == 4
    manifest = json.loads((output / "workload.json").read_text())
    assert manifest["config"]["sampling"] == "sequential"
    assert manifest["config"]["heat_dataset"] is manifest["config"]["heat_field"] is None
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    assert [row["user_id"] for row in rows] == [0, 1, 0, 1]


def test_nosa_long_context_requires_explicit_supported_override():
    arguments = {
        "model": "nosa",
        "num_users": 1,
        "requests": 1,
        "history_tokens": 65536,
        "candidate_tokens": 128,
    }
    with pytest.raises(ValueError, match="exceeds model context"):
        WorkloadConfig(**arguments)
    assert WorkloadConfig(**arguments, context_limit=65664).context_limit == 65664
    assert WorkloadConfig(**arguments, context_limit=262144).context_limit == 262144
    with pytest.raises(ValueError, match="exceeds model context"):
        WorkloadConfig(**arguments, context_limit=65536)
    with pytest.raises(ValueError, match="backend limit"):
        WorkloadConfig(**arguments, context_limit=262145)
    with pytest.raises(ValueError, match="backend limit"):
        WorkloadConfig(
            model="deepseek_v32",
            num_users=1,
            requests=1,
            context_limit=MODEL_FORMATS["deepseek_v32"].MAX_INPUT_TOKENS + 1,
        )


def test_gr_constructor_context_override_preserves_default_boundary(tokenizer):
    text_config = TextConfig(
        user_lengths=(65536,),
        user_probabilities=(1,),
        item_lengths=(128,),
        item_probabilities=(1,),
        max_input_tokens=65664,
    )
    population = HeatPopulation({0: 1.0}, {})
    with pytest.raises(ValueError, match="32768-token context"):
        InputGenerator(population, tokenizer, text_config=text_config)
    with pytest.raises(ValueError, match="32768-token context"):
        create_input_generator(tokenizer=tokenizer, num_users=1, text_config=text_config)
    direct = InputGenerator(population, tokenizer, text_config=text_config, context_limit=65664)
    factory = create_input_generator(
        tokenizer=tokenizer, num_users=1, text_config=text_config, context_limit=65664
    )
    assert direct.context_limit == factory.context_limit == 65664
    assert direct.lengths_for_user(0) == factory.lengths_for_user(0) == (65536, 128)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="context_limit"):
            InputGenerator(population, tokenizer, text_config=text_config, context_limit=invalid)
    with pytest.raises(ValueError, match="65536-token context"):
        InputGenerator(population, tokenizer, text_config=text_config, context_limit=65536)
