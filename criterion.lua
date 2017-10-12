require 'nn'
require 'criterion_filter' 
require 'criterion_ignore'
require 'models/MaskZeroCriterion'
require 'models/IgnoreLabel'
x=torch.Tensor(3,3)
x[1][1]=0.1
x[1][2]=0.8
x[1][3]=0.1
x[2][1]=0.1
x[2][2]=0.8
x[2][3]=0.1
x[3][1]=0.1
x[3][2]=0.1
x[3][3]=0.8
y=torch.Tensor(3)
y[1]=1
y[2]=2
y[3]=3
lll = false
if lll then
    a=torch.load('tmp/a.t7')
    x=a[1]:double()
    y=a[2]:double()
    y=y:resize(y:size(1))
    print(x)
    print(y)
end
w=torch.Tensor{0.840164474,
0.015289448,
0.01322186,
0.012904991,
0.012500848,
0.010241442,
0.008925935,
0.008502364,
0.007041067,
0.006963276,
0.006930536,
0.00636323,
0.006150134,
0.005998818,
0.005986277,
0.005132864,
0.004968185,
0.004451686,
0.003620991,
0.003402087,
0.002400339,
0.002288754,
0.001764469,
0.001759663,
0.001553608,
0.001472656}

-- print(w)

w=torch.Tensor{0.1,0.9}

criterion = nn.ClassNLLCriterion()
e = criterion:forward(x,y)
criterion:backward(x,y)
print("3 class NLL:\t" .. e)
print(criterion.gradInput)
criterion = criterion_filter.Single(criterion, 1)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("terion_filt:\t" .. e)
print(criterion.gradInput)
--y[3]=0
--print("class is labeled as 0.")
-- criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)
criterion = IgnoreLabel(nn.ClassNLLCriterion(),1)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("IgnoreLabel:\t" .. e)
print(criterion.gradInput)

weight=torch.Tensor{0,0.5,0.5}
criterion = nn.ClassNLLCriterion(weight)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("3 0-weight:\t" .. e)
print(criterion.gradInput)

--criterion = nn.ClassNLLCriterion()
--criterion = criterion_filter.Parallel():add(criterion, 1,1)
--e = criterion:forward(x,{y})
--print("2 class NLL:\t" .. e)

--criterion = nn.ParallelIgnoreCriterion():add{
--            criterion = nn.ClassNLLCriterion(),
--            ignore = 1
--}
--e = criterion:forward(x,y)  
--print("2 class NLL:\t" .. e) 

print("====================CrossEntropy====================")
if lll then
    y=a[2]:double() 
end
criterion=nn.CrossEntropyCriterion()
e = criterion:forward(x,y)
criterion:backward(x,y)
print("3 class CEC:\t" .. e)
print(criterion.gradInput)

criterion=nn.CrossEntropyCriterion() 
criterion = criterion_filter.Single(criterion, 1)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("erion_filte:\t" .. e)
print(criterion.gradInput)

criterion = IgnoreLabel(nn.CrossEntropyCriterion(),1)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("IgnoreLabel:\t" .. e)
print(criterion.gradInput)

-- criterion = criterion_filter.Parallel():add(criterion, 1,1)
-- e = criterion:forward(x,{y})
-- print("Pa Ignore Cross:\t" .. e)

criterion = nn.ParallelIgnoreCriterion():add{
            criterion = nn.CrossEntropyCriterion(),
            ignore = 1
}
e = criterion:forward(x,y)  
criterion:backward(x,y)
print("ParallelIgno:\t" .. e)
print(criterion.gradInput)

weight=torch.Tensor{0,1,1}
criterion = nn.CrossEntropyCriterion(weight)
e = criterion:forward(x,y)
criterion:backward(x,y)
print("3 0-wei CEC:\t" .. e)
print(criterion.gradInput)
