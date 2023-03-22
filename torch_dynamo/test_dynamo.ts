Before TExprFuser: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %x.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.1) # test_nnc.py:6:8
  return (%y.1)

After removing redundant profile nodes: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %x.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.1) # test_nnc.py:6:8
  return (%y.1)
After creating fusion groups: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::TensorExprGroup_0(%b.1)
  return (%y.2)
with prim::TensorExprGroup_0 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
After inlining small fusion groups: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::TensorExprGroup_0(%b.1)
  return (%y.2)
with prim::TensorExprGroup_0 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
buildShapeExpressions for 
graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
After guarding fusion groups: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %23 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu), %24 : bool = prim::TypeCheck[types=[Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)]](%b.1)
  %25 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::If(%24)
    block0():
      %y.6 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::TensorExprGroup_0(%23)
      -> (%y.6)
    block1():
      %y.2 : Tensor = prim::FallbackGraph_1(%b.1)
      -> (%y.2)
  %20 : int[] = aten::size(%b.1)
  %21 : int[] = aten::size(%25)
  %22 : int[] = prim::BroadcastSizes(%20, %20)
  return (%25)
with prim::TensorExprGroup_0 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
with prim::FallbackGraph_1 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Tensor = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Tensor = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
After TExprFuser: 
graph(%a.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %5 : int = prim::Constant[value=1]() # test_nnc.py:4:36
  %4 : NoneType = prim::Constant()
  %3 : int[] = prim::Constant[value=[1, 1]]()
  %2 : int[] = prim::Constant[value=[0, 0]]()
  %1 : int[] = prim::Constant[value=[1, 1, 1, 1]]()
  %6 : Float(1, 1, 1, 1, strides=[1, 1, 1, 1], requires_grad=0, device=cpu) = aten::randn(%1, %4, %4, %4, %4) # test_nnc.py:4:24
  %b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::conv2d(%a.1, %6, %4, %3, %2, %3, %5) # test_nnc.py:4:8
  %23 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu), %24 : bool = prim::TypeCheck[types=[Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)]](%b.1)
  %25 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::If(%24)
    block0():
      %y.6 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = prim::TensorExprGroup_0(%23)
      -> (%y.6)
    block1():
      %y.2 : Tensor = prim::FallbackGraph_1(%b.1)
      -> (%y.2)
  return (%25)
with prim::TensorExprGroup_0 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu) = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)
with prim::FallbackGraph_1 = graph(%b.1 : Float(1, 1, 128, 128, strides=[16384, 16384, 128, 1], requires_grad=0, device=cpu)):
  %x.2 : Tensor = aten::mul(%b.1, %b.1) # test_nnc.py:5:8
  %y.2 : Tensor = aten::sin(%x.2) # test_nnc.py:6:8
  return (%y.2)