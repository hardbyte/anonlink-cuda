
import anonlink
import anonlink.concurrency


def chunk(input_a, input_b):
    # Aiming for 2 GiB of results per chunk
    chunk_size_aim = 2 * 2**30/32
    print(chunk_size_aim)
    dataset_sizes = [len(input_a) // 32, len(input_b) // 32]

    chunks = anonlink.concurrency.split_to_chunks(chunk_size_aim, dataset_sizes=dataset_sizes)
    for i, chunk in enumerate(chunks):
        yield chunk




