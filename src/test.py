import torch

if __name__ == "__main__":
    t = torch.tensor([1.], requires_grad=True)
    t.retain_grad()
    a = torch.tensor([2.], requires_grad=True)
    a.retain_grad()
    y = (a * t)
    #y.retain_grad()

    y.backward()
    print(t.grad)
    print(a.grad)
    # print(y.grad)