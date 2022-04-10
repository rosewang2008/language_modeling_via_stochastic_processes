
# how data was processed

* used `en_city` data from WikiSection dataset

`data[i]` structure: `dict_keys(['id', 'type', 'title', 'abstract', 'text', 'annotations'])`

* Check it contains all keys: History, Geography, Demographics, Education
* * If yes, add that entry into the new json dataset
