import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def train_resnet50():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 数据集加载
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_val, download=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"dataloader memory allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, cached:{(torch.cuda.memory_cached())/(2**20):.4}MB")
    # 初始化 ResNet-50 模型
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 有 10 个分类
    model = model.to(device)
    print(f"model ready memory allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, cached:{(torch.cuda.memory_cached())/(2**20):.4}MB")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练参数
    num_epochs = 1
    best_acc = 0.0

    # 训练和验证循环
    step = 0
    print(f"training memory_allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, \
          max_memory_allocated {(torch.cuda.max_memory_allocated())/(2**20):.4} MB, \
          memory_cached:{(torch.cuda.memory_cached())/(2**20):.4}MB, \
          max_memory_cached:{(torch.cuda.max_memory_cached())/(2**20):.4}MB")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"memory allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, cached:{(torch.cuda.memory_cached())/(2**20):.4}MB")
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            step += 1
            if step % 50 == 0:
                print(f"Training {epoch+1}/{num_epochs}, step {step},\
                      memory_allocated {(torch.cuda.memory_allocated())/(2**20)} MB, memory_cached:{(torch.cuda.memory_cached())/(2**20)}MB")

                print(f"training memory_allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, \
                    max_memory_allocated {(torch.cuda.max_memory_allocated())/(2**20):.4} MB, \
                    memory_cached:{(torch.cuda.memory_cached())/(2**20):.4}MB, \
                    max_memory_cached:{(torch.cuda.max_memory_cached())/(2**20):.4}MB")
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # # 验证阶段
        # model.eval()
        # running_corrects = 0
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         _, preds = torch.max(outputs, 1)
        #         running_corrects += torch.sum(preds == labels.data)

        # epoch_acc = running_corrects.double() / len(val_loader.dataset)
        # print(f"Validation Accuracy: {epoch_acc:.4f}")

        # # 保存最佳模型
        # if epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     torch.save(model.state_dict(), "best_resnet50.pth")

    print(f"Best Validation Accuracy: {best_acc:.4f}")


def train_with_cudagraph():

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 数据集加载
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_val, download=True)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"dataloader memory allocated {(torch.cuda.memory_allocated())/(2**20)} MB, cached:{(torch.cuda.memory_cached())/(2**20)}MB")
    # 初始化 ResNet-50 模型
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 有 10 个分类
    model = model.to(device)
    print(f"model ready memory allocated {(torch.cuda.memory_allocated())/(2**20)} MB, cached:{(torch.cuda.memory_cached())/(2**20)}MB")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练参数
    num_epochs = 1
    best_acc = 0.0

    # # 训练和验证循环
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch+1}/{num_epochs}")

    #     # 训练阶段
    #     model.train()
    #     running_loss = 0.0
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item() * inputs.size(0)

    #     epoch_loss = running_loss / len(train_loader.dataset)
    #     print(f"Training Loss: {epoch_loss:.4f}")

    #     # 验证阶段
    #     model.eval()
    #     running_corrects = 0
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, preds = torch.max(outputs, 1)
    #             running_corrects += torch.sum(preds == labels.data)

    #     epoch_acc = running_corrects.double() / len(val_loader.dataset)
    #     print(f"Validation Accuracy: {epoch_acc:.4f}")

    #     # 保存最佳模型
    #     if epoch_acc > best_acc:
    #         best_acc = epoch_acc
    #         torch.save(model.state_dict(), "best_resnet50.pth")

    # print(f"Best Validation Accuracy: {best_acc:.4f}")


    # Warm-up data and target tensors
    example_data, example_target = next(iter(train_loader))
    example_data, example_target = example_data.to(device), example_target.to(device)
    
    # CUDA Graph setup
    static_graph = None

    # Static input and output tensors for graph
    print(f"before static input memory_allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, memory_cached:{(torch.cuda.memory_cached())/(2**20):.4}MB")
    mem_a = torch.cuda.memory_allocated()
    static_data = torch.empty_like(example_data)
    static_target = torch.empty_like(example_target)
    static_loss = torch.empty(1, device=device)
    mem_b = torch.cuda.memory_allocated()
    print(f"static input allocated delta {(mem_b - mem_a)/(2**20)} MB")
    print(f"static input memory_allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, memory_cached:{(torch.cuda.memory_cached())/(2**20):.4}MB")
    # Capture CUDA Graph
    g = torch.cuda.CUDAGraph()

    # Warm-up and capture graph
    model.train()
    for i, (data, target) in enumerate(train_loader):
        if i == 0:
            # Warm-up the model and optimizer
            data, target = data.to(device), target.to(device)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(data)
                    loss = criterion(output, target).unsqueeze(0)
                    loss.backward()
                    optimizer.step()
            torch.cuda.current_stream().wait_stream(s)
            
            static_data.copy_(data)
            static_target.copy_(target)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(g):
                output = model(static_data)
                static_loss = criterion(output, static_target).unsqueeze(0)
                static_loss.backward()
                optimizer.step()
        else:
            break
   
    print(f"graph capture done memory allocated {(torch.cuda.memory_allocated())/(2**20)} MB, cached:{(torch.cuda.memory_cached())/(2**20)}MB")
    print(f"training memory_allocated {(torch.cuda.memory_allocated())/(2**20):.4} MB, \
          max_memory_allocated {(torch.cuda.max_memory_allocated())/(2**20):.4} MB, \
          memory_cached:{(torch.cuda.memory_cached())/(2**20):.4}MB, \
          max_memory_cached:{(torch.cuda.max_memory_cached())/(2**20):.4}MB")
    # Training loop with CUDA Graph
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        print(f"memory_allocated {(torch.cuda.memory_allocated())/(2**20)} MB, memory_cached:{(torch.cuda.memory_cached())/(2**20)}MB")
        for i, (data, target) in enumerate(train_loader):
            # Copy new data to static tensors
            static_data.copy_(data.to(device))
            static_target.copy_(target.to(device))

            # Replay the graph
            g.replay()

            # # Print loss for debugging (optional)
            # if i % 50 == 0:
            #     print(f"Batch {i}, Loss: {static_loss.item()}")
            step += 1
            if step % 50 ==0:
                print(f"Training {epoch+1}/{num_epochs}, step {step}, Loss:{static_loss.item():.4}, \
                    memory_allocated {(torch.cuda.memory_allocated())/(2**20)} MB, memory_cached:{(torch.cuda.memory_cached())/(2**20)}MB")
            
if __name__ == "__main__":
    train_resnet50()
    #train_with_cudagraph()
