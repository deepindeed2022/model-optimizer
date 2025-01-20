with open("perf.txt", "r") as fd:
    lines = fd.readlines()
    print("|Model Name|Format|BatchSize|Inference Time(ms)|")
    print("|:----|:----|:----|:----|")
    
    for line in lines:
        line_data = line.strip().split(" - ")[1].split()
        modelname = line_data[0]
        batch = line_data[1].strip("batch_size=")
        time_val = line_data[3]
        format = modelname.split(".")[-1]
        print(f"|{modelname}|{format}|{batch}|{time_val}|")
        
    