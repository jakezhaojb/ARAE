--[[
  Data loader for MNIST
--]]

require "nn"
torch.setdefaulttensortype("torch.FloatTensor")
require "image"
paths.dofile("aux.lua")

local loader = torch.class("MNISTLoader")

--[[ constructor ]]--
function loader:__init(config)
   -- config parsing
   self.config = deepcopy(config)
   self.nC = self.config.nC or 1
   self.h = 28
   self.w = 28
   -- dataset loading
   self.train = torch.load("/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/train_28x28.th7")
   self.test = torch.load("/misc/vlgscratch3/LecunGroup/michael/datasets/mnist/test_28x28.th7")
   -- preprocessing
   self.train.data = self.train.data:float()
   self.test.data  = self.test.data:float()
   self.train.data:div(255/2):add(-1):clamp(-0.995,0.995)
   self.test.data:div(255/2):add(-1):clamp(-0.995,0.995)
   -- tensor buffers cacheing the output
   self.output = torch.Tensor()
   self.labels = torch.ByteTensor()
   collectgarbage()
end

--[[ shuffle set ]]--
function loader:shuffle(set)
   local this_set
   if set == "train" then
      this_set = self.train
   elseif set == "labeled" then
      this_set = self.labeled
   else
      error("MnistLoader:shuffle(set): set has to be [train|labeled]")
   end
   local randperm = torch.randperm(this_set.data:size(1)):long()
   this_set.data = this_set.data:index(1, randperm)
   this_set.labels = this_set.labels:index(1, randperm)
   collectgarbage()
end

--[[ get one batch ]]--
function loader:getBatch(batchsize, set)
   local this_set
   if set == "train" then
      this_set = self.train
   elseif set == "test" then
      this_set = self.test
   elseif set == "labeled" then
      this_set = self.labeled
   end
   -- preparing buffers
   self.output:resize(batchsize, self.nC, self.h, self.w)
   self.labels:resize(batchsize)
   -- boostrapping samples
   for i = 1, batchsize do
      local idx = torch.random(this_set.data:size(1))
      self.output[i]:copy(this_set.data[idx])
      self.labels[i] = this_set.labels[idx]
   end
   return self.output, self.labels
end

--[[ type cast ]]--
function loader:type(typ)
   if typ ~= nil then
      self.output = self.output:type(typ)
      self.labels = self.labels:type(typ)
      collectgarbage()
   end
   return self
end
--[[ Cudaize ]]--
function loader:cuda()
   self:type("torch.CudaTensor")
end

