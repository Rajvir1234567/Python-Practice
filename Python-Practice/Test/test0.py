import os
parent_folder = "."
folders = { "Except" : ["except.py"] , "headAI":["headAI.php"], "market": ["market.json"],
         "new" : ["new.docx"], "try": ["try.txt"]}

for folder,files in folders.items():
    folder_path = os.path.join(parent_folder,folder)
    os.makedirs(folder_path,exist_ok = True)

    for file in files:
        source = os.path.join(parent_folder,file)
        destination = os.path.join(folder_path,file)

        if os.path.exists(source):
            os.rename(source,destination)
            print(f"{file} moved to {folder}")
        else:
            print(f"{file} not found.")