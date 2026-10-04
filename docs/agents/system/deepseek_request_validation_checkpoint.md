# Isolated CPU request validation candidate

Status: the temporary prototype and opt-in production integration passed CPU
exactness and two bounded timing runs, 2026-10-04. Default serving retains the
original Python method. Full serving/GPU acceptance belongs to the combined
gate; cleanup-audit reuse and token-transfer changes remain unselected.

Contract: replace only the owned list snapshot and exact builtin-int /
nonnegative scan inside frozen C9 `PersistentGRRunner._validate`. Preserve
UID/container checks, metadata validation order, arbitrary positive integer
handling, packed encoding/hash/device transfer and all admission behavior.
The native scan records negativity but reports it only after all exact types
pass. Use public CPython C APIs; do not inspect Python 3.12 integer layout.

The measured candidate keeps the exact Python `ids = list(ids)` snapshot and
Python error site. Its native predicate uses `PyLong_CheckExact` for type
identity and `PyLong_AsLongLongAndOverflow` for sign including arbitrary-size
integers. The unchanged Python wrapper remains responsible for request
metadata. A candidate function is derived from the frozen function by
replacing only the strict-type/nonnegative condition.

An initial snapshot-in-C helper was compiled but remains unselected and
unmeasured. Independent CPU review found its type/sign/refcount handling sound,
but moving the snapshot changed invalid-input traceback-local lifetime and
added container-subclass snapshot surface. Root selected the narrower native
predicate instead, preserving those behaviors directly.

Execution sequence:

1. Compile one small C extension under
   `/tmp/deepseek_request_validation_20261004/` using installed `cc` and this
   interpreter's `sysconfig` include path. Record source/binary/compiler/ABI.
2. Compare complete return values, owned-list semantics, exception class/text
   and metadata read order against frozen C9. Reuse existing strict-type,
   overflow and foreign-metaclass cases; add negative-before-invalid-type,
   large integers and container-subclass snapshots. No timing before this gate.
3. Time complete `_validate` calls for H64K+A128, short inputs and malformed
   cases using bounded randomized pairs. Include list snapshot destruction,
   use CPU wall time, retain all samples and report short-input regressions.
4. Report the measured result and source limitations to root. Do not design or
   integrate a production loader unless root selects it after these results.

## Executed candidate and correctness

Temporary sources and evidence are under
`/tmp/deepseek_request_validation_20261004/`. `strict_token_scan.c` is the
predicate-only candidate. `build_scan.py` invokes installed GCC 13.3.0 with
`-O3 -Wall -Wextra -Werror -std=c11 -fPIC -shared` and the active interpreter's
`sysconfig` include directory. This is CPython 3.12.13,
`cpython-312-x86_64-linux-gnu`; no private integer-layout fields are used.
`scan_build.json` records compiler/executable/header identities, exact argv,
Python executable and ABI. No package dependency or production loader was added.

| Artifact | SHA-256 |
| --- | --- |
| `strict_token_scan.c` | `1869ae003ba6b4e7a1cb7296b77a492ffac7547c77e3ff1a72221294e06ffa78` |
| `_strict_token_scan.cpython-312-x86_64-linux-gnu.so` | `d76766c0f14e538f7ee05e79926dec289a68e9312ca66e8ac8b60636481165c0` |
| `screen_01.json` | `472b3a579bf85fbb379e4eaac3a286f160483ee401523d9e9849630c88a684c7` |
| `bench_01.json` | `8f57eb9f862912f4d933c14ad55dbb40823a90ba35a26fab4528b7a2067de70d` |
| `bench_02.json` | `fc50b40f3d621603d391bf07ff941666abfa12e8536892243b457205c5ac23a9` |

The driver imports the frozen C9 `PersistentGRRunner._validate` and constructs
the candidate by replacing only its strict-token condition. An AST check
restores that one condition and requires exact equality with the original
function tree. Thus Python list snapshot, error sites and metadata statements
are unchanged. Every run checks the same oracle, candidate, helper, binary,
build and driver identities before and after execution.

`screen_01` passed 244 focused comparisons, including H64K+A128, strict builtin
types, foreign metaclasses/coercion, negative-before-invalid-type inputs,
arbitrary signed overflow, missing keys, metadata read order and list/tuple
subclass snapshots. Returned lists retain shallow object identity and survive
source-list mutation. Three later packing comparisons preserve successful
encoding and the original prefix/suffix overflow exception types and context.

The frozen existing token-input and persistent-runner CPU tests then ran
unchanged with the candidate method patched only in that process: 53 passed,
one CUDA test deselected. Its collection-time availability query was replaced
with `False`; no CUDA API was invoked. `torch.cuda.is_initialized()` remained
false throughout screen and both timing runs. Logs are `screen_01.stdout.log`
and the separate stderr file. The initial snapshot-in-C artifact remains
unselected, unmeasured and excluded from all these results.

## Complete `_validate` CPU timings

Both runs used `perf_counter_ns`, three warmup blocks per variant, forty
randomized pairs per case, five calls per H64K block and 1,000 calls per short
block. Each call includes the unchanged Python snapshot and its destruction;
errors are caught inside the timed block. Seeds were 20261004 and 20261005.
All 440 raw pairs per run are retained. Total timed-driver wall intervals were
2.717 and 2.709 seconds. No CPU isolation or frequency/affinity control is
claimed. The host reports Intel Xeon Platinum 8558P, 192 logical CPUs, with all
192 CPUs allowed by the process affinity. `environment.json` records these
facts and the declared OMP/MKL thread limits of eight.

Values below are per-call medians in microseconds; speedup is the ratio of the
two medians. H64K tokens use a recorded deterministic vocabulary-range pattern.
The lightweight owner supplies normal context/capacity bounds; optional
history/candidate bounds and their error order are covered in the screen.

| Case | Run 1 baseline / candidate | Run 2 baseline / candidate | Run 2 speedup |
| --- | --- | --- | --- |
| H64K+A128, valid | 1956.104 / 334.297 | 1942.668 / 331.626 | 5.858x |
| Four tokens, valid | 0.921 / 0.489 | 0.917 / 0.494 | 1.856x |
| Two tokens, valid | 0.848 / 0.481 | 0.849 / 0.488 | 1.741x |
| Positive arbitrary-size ints | 0.890 / 0.504 | 0.894 / 0.510 | 1.752x |
| H64K, invalid first type | 154.693 / 154.022 | 151.235 / 150.651 | 1.004x |
| H64K, invalid last type | 1465.753 / 333.544 | 1458.195 / 332.216 | 4.389x |
| H64K, negative first token | 1948.197 / 334.437 | 1939.328 / 333.487 | 5.815x |
| H64K, negative last token | 1951.784 / 335.332 | 1941.907 / 333.685 | 5.820x |
| Short invalid type | 0.631 / 0.383 | 0.625 / 0.383 | 1.631x |
| Short negative token | 0.769 / 0.397 | 0.769 / 0.396 | 1.939x |
| H64K, invalid prefix | 1949.767 / 333.514 | 1938.926 / 333.714 | 5.810x |

The H64K valid case's median paired savings were 1619.273 and 1610.380
microseconds. The first-token type error remains near parity because both
paths still construct/destroy the same full list snapshot before discovering
the first invalid type. No case has a median regression in either run.

This result supports considering the small predicate for production design.
It does not measure packing/hash/device transfer, cache behavior, GPU work or
full serving latency, and must not be subtracted directly from C9 formal or
profile times. Root owns any production selection and subsequent integration
and serving verification.

## Selected production design (before implementation)

Root selected the predicate for narrow production integration. Add
`serving/token_validation.py` and `serving/csrc/token_validation.cpp`; the
`.cpp` source is covered by existing experiment source snapshots. Keep the
Python snapshot and ValueError site in `PersistentGRRunner._validate`, changing
only its predicate call. Bind the prepared predicate in runner construction,
before any timed request and before acquiring backend resources.

The loader will use standard CPython extension loading and an installed host
compiler, with no PyTorch extension, CUDA call or added dependency. A scoped
fingerprint covers both helper sources, fixed compile flags, resolved compiler
tools and Python ABI plus the actual header dependency closure. A per-key lock,
temporary build and atomic publication protect concurrent initializers; cached
binary bytes are checked against their manifest before loading. Expose the
selected backend and retained source/build/binary identity for root's provenance.
No compiler or filesystem work occurs inside the bound predicate.

Preserve the original Python predicate as the setup-time fallback when the
runtime/compiler/build is unavailable. Select once during initialization;
never catch request validation errors to switch implementations. Ordinary
token validation, packing overflow errors and metadata order stay unchanged.
No package/build framework or generic loader is introduced.

CPU verification will cover native/fallback exactness, dependency-fingerprint
changes, cache reuse/concurrent preparation, failed builds without partial
publication and absence of loading on execute. Reuse the 244-case driver and
existing persistent/token-input tests, then measure complete production
`_validate` against frozen C9 with the established two-seed timing boundary.

The initial design would affect every `PersistentGRRunner` caller, including
NOSA. Before acceptance, root narrowed the implementation to explicit opt-in
so existing default measurements retain their original request path. The final
contract and selected-mode evidence follow.

## Production contract and CPU acceptance

`PersistentGRRunner(..., native_token_validation=False)` is the default.
Its `_validate` method remains literally unchanged from frozen C9; default
construction does not call the native loader. Explicit `True` prepares the
helper before backend ownership/allocation. Only successful native setup
assigns `self._validate = self._validate_native`. Unsupported runtime/compiler
or failed build leaves the original method in place and records the reason.
There is no request-time fallback, loader call or provenance I/O.

The native wrapper duplicates the short original validator. An AST assertion
permits only its method name and token condition to differ; Python list
snapshot, Python ValueError, UID checks, metadata order and capacity checks
remain the same. This bounded duplication preserves the exact default request
path. `serving.token_validation.reference` remains independently importable
and pickleable without a compiler or PyTorch import.

The standard-library loader supports Linux CPython builds with the GIL and
direct GNU/Clang C++ drivers. It rejects compiler launchers and custom CXX
flags; the normal sysconfig `-pthread` is supported. The key covers loader,
C++ source, discovered header closure, Python executable/ABI, driver, available
frontend/assembler/linker tools, fixed flags and declared compile environment.
Cache publication uses a per-key file lock and temporary directory; cache
reuse verifies the binary against its manifest. Loaded `module.__file__` must
resolve to that binary and its bytes must match the retained hash. The native
scan holds the GIL and uses no private integer-layout fields.

Each runner exposes `token_validation_identity`, including requested mode,
selected backend, fallback reason and the native build/binary identity when
available. Root wired capture outside the request interval in selected
DeepSeek motivation/profile and official-serving entry points. Default NOSA
uses the original validator and does not gain a new request-time call. Only
explicit opt-in trajectories need this component's performance replacement;
operator/model-only experiments are outside its scope. Root owns those full
serving runs and publication. CPU component timing below is not serving/MFU
evidence and must not be subtracted from old serving times.

Production evidence is under
`/tmp/deepseek_request_validation_production_20261004/`. The driver extracts
the original method from frozen C9 source and checks the unchanged default
AST. It invokes the actual production native method selected by a real runner
constructor. Native source/dependency and loaded binary hashes are verified
before and after every run. `screen_01.json` passed the same 244 complete
validation cases, owned-snapshot checks and three later packing comparisons.
With only the opt-in constructor keyword injected, 62 existing token-input,
persistent and transient-candidate tests passed; one CUDA test was deselected.
The broader default suite passed 187 tests, including all serving CPU tests,
NOSA persistent integration and ownership tests, and 25 helper/loader tests.
The new tests cover retained invalid-input traceback snapshot lifetime,
default/no-compiler behavior, setup fallback, dependency invalidation,
concurrent cache initialization, cache corruption repair and failed build
cleanup. All checks kept CUDA uninitialized. Ruff passed for changed Python
files. Independent read-only review by `graph_profile` found no remaining
loader/predicate blocker after direct-driver restriction; the earlier
prototype semantic review by `graph_validate` also passed.

Both production timing runs use three warmup blocks, 40 randomized pairs for
each of 11 cases, five calls per H64K block and 1,000 calls per short block.
Seeds remain 20261004/20261005. Each timed call uses the constructor-selected
bound production method and includes list snapshot destruction. Method
binding occurs before the timed block for both versions. Setup/compiler work
is excluded. All 440 raw pairs per run are retained; timed-driver wall
intervals are 2.702 and 2.716 seconds. Hardware, interpreter and uncontrolled
CPU scheduling/frequency boundary match the prototype above.

| Case | Run 1 baseline / native, us | Run 2 baseline / native, us |
| --- | --- | --- |
| H64K+A128, valid | 1949.932 / 332.111 | 1952.584 / 336.133 |
| Four tokens, valid | 0.852 / 0.434 | 0.877 / 0.443 |
| Two tokens, valid | 0.781 / 0.428 | 0.807 / 0.437 |
| Positive arbitrary-size ints | 0.828 / 0.457 | 0.845 / 0.451 |
| H64K, invalid first type | 154.071 / 153.051 | 154.935 / 154.001 |
| H64K, invalid last type | 1465.635 / 330.932 | 1472.468 / 334.710 |
| H64K, negative first token | 1948.866 / 334.040 | 1954.875 / 337.539 |
| H64K, negative last token | 1951.584 / 331.724 | 1955.784 / 337.459 |
| Short invalid type | 0.626 / 0.380 | 0.640 / 0.381 |
| Short negative token | 0.766 / 0.392 | 0.778 / 0.393 |
| H64K, invalid prefix | 1949.529 / 334.207 | 1960.650 / 336.461 |

H64K+A128 median ratios are 5.871x and 5.809x; median paired savings are
1619.664 and 1616.696 us. No case has a median regression in either run.

| Production artifact | SHA-256 |
| --- | --- |
| `serving/persistent.py` | `c4bfb300e4ef98d0d7bb94825d26770c4d2b600eb430549b726a1941b78ae572` |
| `serving/token_validation.py` | `34227128afb5ec52bf57d47c7b7bf8b250cd6da3ba676d20dd3096a7f36a0fde` |
| `serving/csrc/token_validation.cpp` | `144f89b7004bd589cf18850a8ac2b0aee29ce7bdf0b6d8fa4db7ee0e02b0d1af` |
| `serving/tests/test_token_validation.py` | `9048820d4dbd9d3ef25946fe58ed8b25c92821f7154ae2f22aa2646cbd550f7f` |
| Loaded native binary | `ed79d3e717a464379b4f638f608853277b65f602ef05ca5b293e3afa6025cd40` |
| `driver.py` | `53cda381b3a7ded1853ad89c32c8a68bac0cb1a97bfbfa824b740f1e480efc45` |
| `screen_01.json` | `6c94cd288df94671227f126639d45ce7fff7ae31a00eac80888ea959030aede0` |
| `bench_01.json` | `5be5d6bdbeb3b6fca4d155afdf50555474e954099628ee880ab56dbdf1cc7f25` |
| `bench_02.json` | `8166ae2d008f69100ee231969216b9fa01188715fb3e72fae46ae94e139b44b4` |
| `default_cpu.stdout.log` | `12037bce953230b33f166aee0703a9debde49866e95b7bb375772a8faa95c2e6` |

Native build fingerprint:
`fb6ed72f2ace18d4307e8425dbcfbe97625b3bf5ed25f479142c3ea75093b728`.
Its binary is the matching key directory under
`/root/.cache/cxldsagr/token-validation/`; the complete resolved path and
header/tool identity remain in each production JSON record.
