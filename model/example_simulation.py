from model import ReadingModel
from model_components import FixationOutput

texts = ['He likes you.']

# initialize model with default config
model = ReadingModel(texts)

# run reading simulation
output:list[list[list[FixationOutput]]] = model.read(output_filepath=f'../data/model_output/{model.time}/example_simulation.csv', number_of_simulations=1, verbose=False)
