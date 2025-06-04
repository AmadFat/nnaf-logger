import math
import random
import time
from argparse import ArgumentParser

from nnaf_logger import LogConfig, Loggerv2, LogLevel, WandbConfig

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    logger = Loggerv2(
        name="test",
        print_info_interval=3,
        log_config=LogConfig(level=getattr(LogLevel, args.mode.upper()), save_for_man=True, save_as_json=True, refresh_dir=True),
        wandb_config=WandbConfig(anonymous="allow", api_key=None, dir=".", refresh_dir=True)
    )   

    for epoch in range(1, 3):
        for step in range(1, 4):
            logger.add("Mock to log loss.", level="train", epoch=epoch, step=step, tag="train", 
                    loss=(1 / (math.log(epoch) + 1) + random.random() * 0.1))
            logger.add("Mock to log aux loss.", level="train", epoch=epoch, step=step, tag="train",
                    auxloss=0.5 * random.random())
            logger.commit("Mock to commit step.", level="train", epoch=epoch, step=step)
            time.sleep(0.2)
        
        logger.test("Mock to test", metric1="value1", metric2="value2")

    logger.close()
