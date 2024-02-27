# class mlp_configs:
#     name = "mlp_spatial"
#     input_size = 6
#     output_size = 1
#     hidden_layers = [2048, 1024, 512, 256, 128, 64]

class mlp_configs:
    name = "mlp_temporal"
    input_size = 8
    output_size = 1
    hidden_layers = [2048, 1024, 512, 256, 128, 64]


class draw_configs:
    T = 10
    batch_size = 64
    z_size = 10
    N = 5
    dec_size = 256
    enc_size = 256
    epoch_num = 20
    lr = 1e-3
    b1 = 0.5
    b2 = 0.999
    clip = 5.0
    network_name = "draw_model"
    split_at = 0.8
    shuffle = True
    drop_last = True
    step_size = 20
    gamma = 0.95
