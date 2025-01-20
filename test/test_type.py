""" I was writing a dataloader from a video stream. I ran some numbers.
# in a nutshell. 
-> np.transpose() or torch.permute() is faster as uint8, no difference between torch and numpy
-> np.uint8/number results in np.float64, never do it, if anything cast as np.float32
-> convert to pytorch before converting uint8 to float32
-> contiguous() is is faster in torch than numpy
-> contiguous() is faster for torch.float32 than for torch.uint8
-> convert to CUDA in the numpy to pytorch conversion, if you can.
-> in CPU tensor/my_float is > 130% more costly than tensor.div_(myfloat), however tensor.div_()
does not keep track of gradients, so be careful using it.
-> tensor.div_(anyfloat) has the same cost of tensor.div_(1.0) even when the second is a trivial operation.

When loading a dataset a quite typical operation is to load the data - which may come thru numpy -
convert it to float, premute it and convert to pytorch.
If one does so naively the time cost of the operation can be almost an order of magnitude slower. 
Conclusions marked with arrows '->'

Tests:
1. a naive way of converting to float would be myndarray/255.
: problem, numpy by default uses float64, this increases the time, 
  then converting float64 to float32, adds more time
  
2. simply making the denominator in numpy a float 32 quadruples the speed of the operation
-> never convert npuint8 to float without typing the denominator as float32

3. changing order of operations, converting to torch, then converting to float32 makes it even faster
-> convert to pytorch before converting from uint8 to float32

Adding another operation, permutation.
4. But it isnt as simple, in torch we typically want to permute the channels from H,W,C to C,H,W
permutation should be followed by a conntiguous() call otherwise, with some operations in torch there
can be some severe cache misses.
In this section I discard the worst of the numpy converstions (uint->float64->torch->float32)  comparing only
variations of (2) and (3) above

-> dividing tensor by number does not incur cache misses
-> permuting is significantly faster as uint8, but it is similar in numpy and pytorch
-> contiguity is faster in pytorch and float32

Adding cuda conversion
5. Of course you probably want this in CUDA / profile isnt valid as cuda isnt synchronized,
in any case it is faster than CPU.

6. Division:
-> if grad does not need to be computed use inplace tensor.div_(x) it is ~ 60% faster than tensor = tensor/x
-> tensor._div(1.0) has the same cost as tensor._div(any number)


"""
import timeit
import numpy as np
import torch
# initialize rgb uint8 image
# h:224 w:224 c:3 
myomy = (np.random.random([224,224,3])*255).astype(np.uint8)

s = "img=np.transpose(myomy, (2, 0, 1)).astype(np.float32) / 255.0;img-=np.array([0.485,0.564,0.406]).reshape(3,1,1);img/=np.array([0.229,0.224,0.225]).reshape(3,1,1)"
ms = timeit.timeit(s, number=10000, globals=globals())
print("npuint8-> transpose -> float -> /255:\t%dus /loop; naive"%(ms*100))
s = "img=np.transpose(myomy, (2, 0, 1)) / 255.0;img-=np.array([0.485,0.564,0.406]).reshape(3,1,1);img/=np.array([0.229,0.224,0.225]).reshape(3,1,1)"
ms = timeit.timeit(s, number=10000, globals=globals())
print("npuint8-> transpose -> /255:\t%dus /loop; naive"%(ms*100)) 
s = "img=np.transpose(np.array(myomy, dtype=np.float32), (2, 0, 1)) / 255.0;;img-=np.array([0.485,0.564,0.406]).reshape(3,1,1);img/=np.array([0.229,0.224,0.225]).reshape(3,1,1)"
ms = timeit.timeit(s, number=10000, globals=globals())
print("npuint8-> float -> transpose -> /255:\t%dus /loop; naive"%(ms*100))
print(np.array(myomy, dtype=np.float32))
# # ---- Operations, uint8->float32, numpy->torch
# # 1/ naive conversion: ~ approx 1 ms
# s = "torch.from_numpy(myomy/255).to(dtype=torch.float)"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> float-> torch ->float32:\t%dus /loop; naive"%(ms*100))
# # npuint8-> float-> torch ->float32:	1161us /loop; naive

# # 2/ diviing by np.float32 makes operations: ~4.5x faster
# s = "torch.from_numpy(myomy/np.array(255, dtype=np.float32))"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> torch-> float32:\t%dus /loop"%(ms*100))
# #npuint8-> float32-> torch:	255us /loop

# # 3/ converting it to torch.uint8 dividing in torch: ~7.5x faster
# s = "torch.from_numpy(myomy).to(dtype=torch.float)/255."
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> float32-> torch:\t%dus /loop"%(ms*100))
# # npuint8-> torch-> float32:	150us /loop

# # ---- Operations, uint8->float32, transpose, contiguous, numpy->torch
# # 4/ Adding transpose and contiguous to this equations, increases the time
# # 4/a this test runs transpose and contiguous in numpy 
# s = "torch.from_numpy(np.ascontiguousarray(myomy.transpose(2,0,1))/np.array(255, dtype=np.float32))"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> transpose -> contiguous -> float32-> TORCH:\t%dus /loop"%(ms*100))
# # npuint8-> transpose -> contiguous -> float32-> TORCH:	507us /loop

# # 4/b this test runs transpose in numpy and contiguous in pytorch. 
# # -> pytorch contiguous is faster than numpy contiguous, even if 
# #    nupmy contiguous is called on uint8
# #    pytorch contiguous is being called on float32
# s = "torch.from_numpy(myomy.transpose(2,0,1)/np.array(255, dtype=np.float32)).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> transpose -> float32-> TORCH -> contiguous:\t%dus /loop"%(ms*100))
# #npuint8-> transpose -> float32-> TORCH -> contiguous:	391us /loop

# # 4/c transpose, to torch, contiguous to float
# s = "torch.from_numpy(myomy.transpose(2,0,1)).contiguous().to(dtype=torch.float)/255."
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> transpose-> TORCH-> contiguous -> float32:\t%dus /loop"%(ms*100))
# # npuint8-> transpose-> TORCH-> contiguous -> float32:	361us /loop

# # 4/d transpose, to torch, to float , contiguous
# # -> this is the fastest of the operations: contiguous as pytorch float32
# # I can only guess the reasons for this:
# #   simply dividing does not cause cache misses
# #   float32 operations are well optimized in torch 1.1
# s = "(torch.from_numpy(myomy.transpose(2,0,1)).to(dtype=torch.float)/255.).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> transpose-> TORCH-> float32-> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> transpose-> TORCH-> float32-> contiguous:	265us /loop

# # 4/e it would stand to reason given the examples above, that doing the permutation operation in pytorch
# # is also faster, so we try 3 variations, 
# # permuting and contiguous at the end of the operations
# s = "(torch.from_numpy(myomy).to(dtype=torch.float)/255.).permute(2,0,1).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> float32-> permute -> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> float32-> permute -> contiguous:	582us /loop

# # 4/f permuting and contiguous as torch.uint8, then convert to float
# s = "(torch.from_numpy(myomy)).permute(2,0,1).contiguous().to(dtype=torch.float)/255."
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> permute -> contiguous -> float32:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> permute -> contiguous -> float32:	352us /loop

# # 4/g permuting  as torch.uint8, then float32 then contiguous
# # -> permuting is significantly faster as uint8, but it is the same in numpy and pytorch
# # -> contiguity is faster in pytorch and float32
# s = "((torch.from_numpy(myomy)).permute(2,0,1).to(dtype=torch.float)/255.).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> permute -> float32 -> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> permute -> float32 -> contiguous:	251us /loop

# # ---- Operations, uint8->float32, transpose, contiguous, numpy->torch, cpu -> cuda
# # 5/ taking the faster of the operations (4/g) and adding CUDA
# # 5/a to CUDA after all conversions
# s = "((torch.from_numpy(myomy)).permute(2,0,1).to(dtype=torch.float)/255.).contiguous().cuda()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> permute -> float32 -> divide-> contiguous -> cuda:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> permute -> float32 -> divide-> contiguous -> cuda:	717us /loop

# # 5/b CUDA before contiguous: minor speed up
# s = "((torch.from_numpy(myomy)).permute(2,0,1).to(dtype=torch.float)/255.).cuda().contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> permute -> float32 -> divide -> cuda -> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> permute -> float32 -> divide -> cuda -> contiguous:	688us /loop

# # 5/c CUDA before divide: better
# s = "((torch.from_numpy(myomy)).permute(2,0,1).to(dtype=torch.float, device= 'cuda')/255.).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> permute -> float32 cuda -> divide -> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> permute -> float32 cuda -> divide -> contiguous:	527us /loo

# # 5/d use cuda as soon as you can  <- this is the best number
# s = "((torch.from_numpy(myomy)).to(device='cuda').permute(2,0,1).to(dtype=torch.float)/255.).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> cuda-> permute -> float32 divide -> contiguous:\t%dus /loop"%(ms*100))
# # npuint8-> TORCH-> cuda-> permute -> float32 divide -> contiguous:	159us /loop


# # 5/e finally just to check, in CUDA permutation is still faster as uint8 than as float32
# s = "((torch.from_numpy(myomy)).to(device='cuda').to(dtype=torch.float)/255.).permute(2,0,1).contiguous()"
# ms = timeit.timeit(s, number=10000, globals=globals())
# print("npuint8-> TORCH-> cuda -> float32 divide-> permute -> contiguous:\t%dus /loop"%(ms*100))
# #npuint8-> TORCH-> cuda -> float32 divide-> permute -> contiguous:	229us /loop

# # 6 division in CPU
# # a/ standard division of a tensor
# tensor = torch.ones([1,3,1024,1024], dtype=torch.float32, device="cpu")
# s = "y = tensor/255."
# ms = timeit.timeit(s, number=1000, globals=globals())
# print("tensor/255.0 \t\t%.3fms /loop"%(ms))
# # tensor/255.0 		    1.299ms /loop

# # b/ using tensor.div_(div) ~ 40% the cost than tensor/div
# # division by any float
# s = "tensor.div_(255.0)"
# ms = timeit.timeit(s, number=1000, globals=globals())
# print("tensor.div_(255.0). \t%.3fms /loop"%(ms))
# # tensor.div_(255.0). 	0.533ms /loop

# # time of tensor.div_(1.0) = time of tensor.div_(255.0)
# s = "tensor.div_(1.0)"
# ms = timeit.timeit(s, number=1000, globals=globals())
# print("tensor.div_(1.0). \t%.3fms /loop"%(ms))
# # tensor.div_(1.0). 	0.510ms /loop

# # funcion returns self
# # one may want to use numpy.equal(a,b, atol=1e-8) or something like this here.
# # in cpu
# xdiv = lambda x, div: x if div==1.0 else x.div_(div)
# s = "xdiv(tensor, 1.)"
# ms = timeit.timeit(s, number=1000, globals=globals())
# print("if False, return self \t%.6fms /loop"%(ms))
# # if False, return self 	0.000049ms /loop
