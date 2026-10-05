import hashlib


def derive_seed(parent_seed: int, name: str) -> int:
    value = f"{parent_seed}:{name}".encode("utf-8")
    digest = hashlib.sha256(value).digest()

    return int.from_bytes(digest[:8], byteorder="big")


def make_run_seeds(master_seed: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("Run count must be positive.")

    return [
        derive_seed(master_seed, f"run:{index}")
        for index in range(count)
    ]