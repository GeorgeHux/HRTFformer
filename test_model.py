from model.model import *

if __name__ == "__main__":
    print("------test model encoder------")
    encoder_config_dict = {
    "in_channels": 256,
    "hidden_size": 4096,
    "num_transformer_layers": 2,
    "num_heads": 8,
    "num_groups": 4,
    "dropout": 0.1,
    "num_initial_coeff": 27,
    "max_num_coeff": 484
    }
    encoder_config = ModelConfig(**encoder_config_dict)
    batch_size = 2
    lr = torch.randn(batch_size, encoder_config.num_initial_coeff, encoder_config.in_channels)
    encoder = Encoder(encoder_config)
    encoder_out = encoder(lr)
    print(encoder_out.shape)

    print("-----test model decoder------")
    decoder_config_dict = {
    "in_channels": 2048,
    "hidden_size": 4096,
    "num_transformer_layers": 2,
    "num_heads": 8,
    "num_groups": 4,
    "dropout": 0.1,
    "num_initial_coeff": 4,
    "max_num_coeff": 484
    }
    decoder_config = ModelConfig(**decoder_config_dict)
    decoder = Decoder(decoder_config)
    decoder_out = decoder(encoder_out)
    print(decoder_out.shape)

    print("-----test final model------")
    hrtf_transformer = HRTF_Transformer(encoder_config, decoder_config)
    sr = hrtf_transformer(lr)
    print(sr.shape)