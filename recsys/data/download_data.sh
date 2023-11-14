wget https://gist.githubusercontent.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e/raw/20ac954e202fe6a038c2b4bb476703c02fe0df87/ya.py

# Download data
python ya.py https://disk.yandex.ru/d/Q9gnaU4HvL5nyQ ./
python ya.py https://disk.yandex.ru/d/_fitwckjDkCIPA ./

# Extract dataset
tar -xzvf data.tar.gz
mkdir track_embeddings
tar -xzvf track_embeddings.tar.gz -C track_embeddings
