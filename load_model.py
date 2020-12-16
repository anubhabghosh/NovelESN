from novel_esn import NovelEsn
from armodel import Linear_AR
from testrnn_aliter import RNN_model

def load_model_with_opts(options, model_type):
    """ This function is used for loading the appropriate model
    with the configuration details mentioned in the 'options' (.json)
    file

    Args:
        options ([json_dict]): dictionary containing the list of options 
        for each model considered
        model_type ([str]): type of model under consideration
    """

    if model_type == "esn":

        model = NovelEsn(
                        num_neur=options[model_type]["num_neurons"],
                        conn_per_neur=options[model_type]["conn_per_neuron"],
                        spectr_rad=options[model_type]["spectral_radius"],
                        tau=options[model_type]["tau"],
                        history_q=options[model_type]["history_q"],
                        history_p=options[model_type]["history_p"],
                        beta_regularizer=options[model_type]["beta_regularizer"]
                        )
    
    elif model_type == "linear_ar":

        model = Linear_AR(
                        num_taps=options[model_type]["num_taps"],
                        lossfn_type=options[model_type]["lossfn_type"],
                        lr=options[model_type]["lr"],
                        num_epochs=options[model_type]["num_epochs"],
                        init_net=options[model_type]["init_net"],
                        device=options[model_type]["device"] 
        )
    elif model_type in ["rnn", "lstm", "gru"]:

        model = RNN_model(
                        input_size=options[model_type]["input_size"],
                        output_size=options[model_type]["output_size"],
                        n_hidden=options[model_type]["n_hidden"],
                        n_layers=options[model_type]["n_layers"],
                        num_directions=options[model_type]["num_directions"],
                        model_type=options[model_type]["model_type"],
                        batch_first=options[model_type]["batch_first"],
                        lr=options[model_type]["lr"],
                        device=options[model_type]["device"],
                        num_epochs=options[model_type]["num_epochs"],
        )
    
    return model