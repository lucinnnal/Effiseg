import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    # ======================== seed ======================== #
    parser.add_argument('--seed', type=int, default=42)

    # ======================== train, test ======================== #
    parser.add_argument('--is_inference', type=bool, default=False)

    # ======================== data ======================== #
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--data_name', type=str, default='mnist')
    parser.add_argument('--bf16', type=bool, default=False)

    # ======================== model ======================== #
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--conv_module', type=str, default='cnn')
    parser.add_argument('--fc_module', type=str, default='linear')
    parser.add_argument('--grid_size', type=int, default=5)

    # ======================== training ======================== #
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)

    # ======================== save ======================== #
    parser.add_argument('--save_dir', type=str, default='model')
    parser.add_argument('--load_dir', type=str, default='model')

    return parser.parse_args()