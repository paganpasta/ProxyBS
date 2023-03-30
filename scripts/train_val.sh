export KEY=44cbeba0f3fb820fa19ad87ba2beab2f4207e6f0
export ENTITY=paganpasta
export PROJECT=ProxyBS

python main_val.py --data cifar10 --data_path /datasets/ --alpha 0.1 --model wrn28 --method val  --group cifar10
python main_val.py --data cifar10 --data_path /datasets/ --alpha 0.05 --model wrn28 --method val  --group cifar10
python main_val.py --data cifar10 --data_path /datasets/ --alpha 0.01 --model wrn28 --method val  --group cifar10
