import pylsl
from mne_lsl.stream import StreamLSL as Stream
import mne

print("looking for streams")

# print stream names
stream_name = '0'
while stream_name != 'EE225-000000-000758-02-DESKTOP-8G8988B':
# while True:
    streams = pylsl.resolve_streams()
    for ii, stream in enumerate(streams):
        stream_name = stream.name()
        print('%d: %s' % (ii, stream_name))

print('found stream')



stream = Stream(bufsize=5, name= stream_name)  # 5 seconds of buffer
stream.connect(acquisition_delay=0.2)
print(stream.info)

stream.pick([0,1,2])
data, ts = stream.get_data(20)

raw = mne.io.RawArray(data=data, info=stream.info, verbose=False)
#raw = stream.to_mne()
print(raw)
print(raw.ch_names)


