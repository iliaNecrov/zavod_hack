# zavod_hack

# Команда Своего рода учёные представляет реализацию решения задачи: "Суммаризация комментариев в социальных медиа"

## Запуск решения
1) Установка зависимостей
Для запуска решения необходимо установить библиотеки из активированной виртуальной среды:
```shell
pip install -r dependencies.txt
```
2) Переместите свой датасет в папку data (в этой же папке будет и output), папка data обязательно должна находится в корне проекта, откуда будет произведен запуск solution.py скрипта (в общем также как и на гитхабе)
3) Запустите solution.py слеюущей командой:
   - Для всех комменатариев:
     ```shell
     python solution.py -st all_comments -i dataset.jsonl -o result.jsonl
     ```
   - Для комментарией с прямым отношением к посту:
     ```shell
     python solution.py -st post_comments -i dataset.jsonl -o result.jsonl
     ```
   - Для комментарией с косвенным отношением к посту:
     ```shell
     python solution.py -st topic_comments -i dataset.jsonl -o result.jsonl
     ```

   где ```dataset.jsonl``` и ```result.jsonl``` названия датасета, который вы положили в ```data``` и датасет, который сохранится также в папке ```data``` с суммаризацией.
4) Заберите готовый файл из папки ```data```
