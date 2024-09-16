import logging

def setup_logging():
    logging.basicConfig(
        filename='logs/train_logs.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
