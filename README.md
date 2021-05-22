# DeepRL-Pong
Deep Reinforcement Learning bot playing Pong game.


--------------
![pong-v0](images/pong-v0.png)

--------------
## Quick start

Nice virtualenv tutorial [here](https://computingforgeeks.com/fix-mkvirtualenv-command-not-found-ubuntu/)
```bash
which python3.8
mkvirtualenv -p <path to python3> <name>
workon <name>
```

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

[OpenAI Gym](https://github.com/openai/gym)

```
git clone https://github.com/openai/gym
cd gym && pip install -e .
```
or
```
pip install gym
pip install gym[atari]
```

Downloading ROM (https://github.com/openai/atari-py#roms):
* Download it from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html

```bash
unrar x Roms.rar
unzip ROMS.zip 
python -m atari_py.import_roms ROMS
```

Now you can run:
```bash
python gym_demo.py
```