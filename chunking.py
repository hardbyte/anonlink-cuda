
import anonlink
import anonlink.concurrency


def chunk(input_a, input_b):
    # Aiming for >2 GiB of results per chunk
    chunk_size_aim = 2 * 2**30 * 128

    dataset_sizes = [len(input_a), len(input_b)]
    # size in number of primitive uint32 elements

    chunks = anonlink.concurrency.split_to_chunks(chunk_size_aim, dataset_sizes=dataset_sizes)
    for i, chunk in enumerate(chunks):
        yield chunk




