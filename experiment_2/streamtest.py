import pylsl


print("looking for streams")

# print stream names
while True:
    streams = pylsl.resolve_streams()
    for ii, stream in enumerate(streams):
        print('%d: %s' % (ii, stream.hostname()))