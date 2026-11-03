"""Скачивает Beans с Hugging Face и сохраняет картинки по папкам классов."""

import argparse
import csv
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DATASET = 'AI-Lab-Makerere/beans'
REVISION = '27aa014ce09b193e1a6f58112d4a66e0eddb69c5'


def prepare(output):
    if (output / 'labels.csv').exists():
        raise ValueError('Данные уже подготовлены. Для повторной подготовки укажите новую папку.')
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET, revision=REVISION, cache_dir=str(output / 'raw'))
    classes = dataset['train'].features['labels'].names
    rows = []
    counts = {}
    for source_split, local_split in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        per_class = [0] * len(classes)
        for index, sample in enumerate(dataset[source_split]):
            label = sample['labels']
            # Для отдельного теста собираем по 20 изображений каждого класса.
            if local_split == 'test' and per_class[label] >= 20:
                continue
            image = sample['image'].convert('RGB').resize((96, 96), Image.Resampling.BILINEAR)
            relative = Path(local_split) / classes[label] / f'leaf_{index:04d}.jpg'
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=95)
            rows.append([relative.as_posix(), local_split, classes[label], source_split, index])
            per_class[label] += 1
        counts[local_split] = per_class
        print(f'{local_split}: {sum(per_class)} изображений; по классам {per_class}', flush=True)
    if counts['test'] != [20] * len(classes):
        raise ValueError('Для теста нужно по 20 изображений каждого класса')
    with (output / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'split', 'class', 'source_split', 'source_index'])
        writer.writerows(rows)
    metadata = {'dataset': DATASET, 'revision': REVISION, 'classes': classes, 'counts': counts,
                'license': 'MIT', 'image_size': 96}
    (output / 'source.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Подготовка фотографий листьев из Hugging Face Beans')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'data')
    args = parser.parse_args()
    try:
        prepare(args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
