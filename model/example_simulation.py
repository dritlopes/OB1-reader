from model import ReadingModel

text = ("There are now rumblings that Apple might soon invade the smart watch space, though the company is maintaining its customary silence. "
        "The watch doesn't have a microphone or speaker, but you can use it to control the music on your phone. "
        "You can glance at the watch face to view the artist and title of a song.")

# initialize model with default config
model = ReadingModel([text])

# run reading simulation
output = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv')

