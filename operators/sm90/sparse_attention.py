"""Reserved SM90 entrypoint for overlapping sparse attention and KV fetching."""


class SM90SparseAttention:
    """Future operator consumes logical selection and cache access together.

    No eager gather or all-records-resident contract is imposed here. This is an
    interface reservation, not a GPU implementation or a validated SM90 path.
    """

    def __call__(self, q, selection, cache_access, context):
        raise NotImplementedError(
            "SM90 sparse attention / offloaded cache fetch is not implemented"
        )
