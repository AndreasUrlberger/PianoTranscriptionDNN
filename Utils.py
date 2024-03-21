
def write_file_completely(file, content):
    bytes_to_write = len(content)
    bytes_written = 0
    while bytes_written < bytes_to_write:
        bytes_written += file.write(content[bytes_written:])