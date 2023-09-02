import torch
x = torch.rand(100)
x = x.to(torch.device('cuda'))
if torch.cuda.is_available():
	print ("Yes")
else:
	print ("No")

device = torch.device("cuda:0")
