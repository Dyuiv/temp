import json


def load_ids_from_json(file_path):
    """Загружает JSON и возвращает множество ID."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {item['id'] for item in data}
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return set()


def compare_json_ids(file1, file2):
    """Сравнивает ID в двух JSON-файлах и выводит статистику."""
    ids1 = load_ids_from_json(file1)
    ids2 = load_ids_from_json(file2)

    print(f"Уникальные ID в {file1}: {len(ids1)}")
    print(f"Уникальные ID в {file2}: {len(ids2)}")

    common_ids = ids1 & ids2
    unique_in_file1 = ids1 - ids2
    unique_in_file2 = ids2 - ids1

    print(f"\nОбщие ID (есть в обоих файлах): {len(common_ids)}")
    print(f"Уникальные для {file1}: {len(unique_in_file1)}")
    print(f"Уникальные для {file2}: {len(unique_in_file2)}")

    # Дополнительно: можно сохранить результаты в файл
    result = {
        "file1": file1,
        "file2": file2,
        "unique_in_file1": len(ids1),
        "unique_in_file2": len(ids2),
        "common_ids": len(common_ids),
        "unique_ids_file1": len(unique_in_file1),
        "unique_ids_file2": len(unique_in_file2),
        "common_ids_list": list(common_ids),
        "unique_ids_file1_list": list(unique_in_file1),
        "unique_ids_file2_list": list(unique_in_file2)
    }

    with open('comparison_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nПодробные результаты сохранены в comparison_result.json")


if __name__ == "__main__":
    compare_json_ids('data.json', 'data2.json')