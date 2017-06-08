--[[
  Generate file
--]]

require 'cutorch' 
require 'cunn' 
require 'cudnn' 
require 'nngraph'
require 'dpnn'
paths.dofile("aux.lua")
paths.dofile("mnist.lua")

-------------- Setting ----------------
local cmd = torch.CmdLine()
cmd:option('--imgname', 'output.png', 'save image name')
cmd:option('--modelpath', '', 'model path')
cmd:option('--nrow', 10, 'nrow')
local opt_eval = cmd:parse(arg or {})

-------------- Model reloading ----------------
local loaded = torch.load(opt_eval.modelpath)
local model, opt = unpack(loaded)
local AE = model:get(1)
local AE_ = nn.Sequential():add(AE:get(1)):add(AE:get(3))
local D = model:get(2)
local G = model:get(3)
model:evaluate()

-------------- Data ----------------
dataloader = MNISTLoader(opt)
dataloader:cuda()
function binarize(x, res)
   local maxval = 0.995
   local minval = -0.995
   res:resizeAs(x):copy(x:gt((maxval+minval)/2))
   return res
end
local x_binmnist_in = torch.CudaTensor()

-------------- Generation---------------
--[[ generate fake samples ]]--
local noise = torch.CudaTensor(opt.batchSize, opt.noiseDim)
noise:normal()
local fake_hid = G:forward(noise)
local fake_gen = AE_:get(2):forward(fake_hid):float()
local _, fake_gen_max = fake_gen:max(2)
local fake_gen_max = fake_gen_max:mul(2):add(-3)
local irec = image.toDisplayTensor({input=fake_gen_max, nrow=opt_eval.nrow, padding=1})

--[[ generate real samples ]]--
local x_mnist = dataloader:getBatch(opt.batchSize, "test")
x_binmnist_in = binarize(x_mnist, x_binmnist_in)
x_binmnist_in:mul(2):add(-1)
local x_ = AE_:forward(x_binmnist_in)
local _,x_ = x_:max(2)
local x_ = x_:mul(2):add(-3)
local irec0 = image.toDisplayTensor({input=x_, padding=1, nrow=opt_eval.nrow})
local splitbar = torch.Tensor(1,4,irec0:size(2)):fill(0.5)

--[[ concatnation ]]--
local todisp = torch.cat({irec0,splitbar,irec},2)

--[[ save ]]--
image.save(opt_eval.imgname, todisp)
