--[[
  Main file for training binarized-MNIST ARAE.
--]]

require 'cutorch' 
require 'cunn' 
require 'cudnn' 
require 'nngraph'
require 'optim'
require 'dpnn'
paths.dofile("aux.lua")
paths.dofile("mnist.lua")

---------------------------------------
-------------- Setting ----------------
---------------------------------------
local cmd = torch.CmdLine()
-- general options
cmd:option('--devid', 1, 'gpu id')
cmd:option('--savename', 4001, 'savename of the save')

-- training settings
cmd:option('--nEpoches', 100, 'Number of "epoches"')
cmd:option('--nIters', 100, 'Number of "iteration"/"minibatches" in each epoch')
cmd:option('--nItersAE', 1, 'Number of iters trained on AE within each iteration')
cmd:option('--nItersGAN', 1, 'Number of iters trained on GAN within each iteration')
cmd:option('--batchSize', 100, 'Batch size')
cmd:option('--learningRateAE', 0.0005, 'Learning rate for auto-encoder')
cmd:option('--learningRateG', 0.0005, 'Learning rate for generator')
cmd:option('--learningRateD', 5e-05, 'Learning rate for discriminator')
-- model settings
cmd:option('--noiseDim', 32, 'Dimension of noise vector')
cmd:option('--hidDim', 100, 'Dimension of code vector')
cmd:option('--archEnc', '800-400', 'architecture of encoder')
cmd:option('--archDec', '400-800-1000', 'architecture of decoder')
cmd:option('--archG', '64-100-150', 'architecture of G')
cmd:option('--archD', '100-60-20', 'architecture of D')
cmd:option('--noiseAE', 0.4, 'std of the noise added to the code vector')
cmd:option('--noiseAnne', 0.99, 'noise exponential decay factor')
-- gan settings
cmd:option('--nItersD', 10, 'Number of iterations for training WGAN critic')
cmd:option('--clamp', 0.05, 'WGAN critic clamp')
cmd:option('--gan2enc', -0.2, 'Multiplier to the gradient from GAN to enc')

local opt = cmd:parse(arg or {})
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.devid)

-- logging aux
local filename = paths.concat(opt.savename, "log")
os.execute("mkdir -p " .. opt.savename)
print("Saving folder name: " .. opt.savename)
function print_(cont, to_stdout)
   local to_stdout = (to_stdout==nil) and true or to_stdout
   local str
   if type(cont) == "table" then
      str = table.concat(cont, '\n')
   elseif type(cont) == "string" then
      str = cont
   end
   if to_stdout then
      print(str)
   end
   file = io.open(filename, 'a')
   file:write(str .. "\n")
   file:close()
end

---------------------------------------
-------------- Data -------------------
---------------------------------------
dataloader = MNISTLoader(opt)
dataloader:cuda()
function binarize(x, res)
   local maxval = 0.995
   local minval = -0.995
   res:resizeAs(x):copy(x:gt((maxval+minval)/2))
   return res
end
-- buffers for discretetion
local x_binmnist_in = torch.CudaTensor()
local x_binmnist_ou = torch.CudaTensor()
-- data preparation
function prep_batch(set)
   local set = set or "train"
   local x_mnist = dataloader:getBatch(opt.batchSize, set)
   x_binmnist_in = binarize(x_mnist, x_binmnist_in)
   x_binmnist_in:mul(2):add(-1)
   x_binmnist_ou = binarize(x_mnist, x_binmnist_ou):squeeze()
   x_binmnist_ou:add(1)
end

---------------------------------------
-------------- Model ------------------
---------------------------------------
--[[ AE ]]--
AE = nn.Sequential()
local arch_enc = convert_option(opt.archEnc)
local Enc = nn.Sequential()
local Dec = nn.Sequential()
AE:add(Enc)
-- encoder
Enc:add(nn.View(784):setNumInputDims(3))
table.insert(arch_enc, 1, 784)
for i = 1, #arch_enc-1 do
   Enc:add(nn.Linear(arch_enc[i],arch_enc[i+1]))
   Enc:add(nn.BatchNormalization(arch_enc[i+1]))
   Enc:add(cudnn.ReLU())
end
Enc:add(nn.Linear(arch_enc[#arch_enc], opt.hidDim))
   Enc:add(nn.Normalize(2))
AE:add(nn.WhiteNoise(0, opt.noiseAE))
-- decoder
AE:add(Dec)
local arch_dec = convert_option(opt.archDec)
table.insert(arch_dec, 1, opt.hidDim)
for i = 1, #arch_dec-1 do
   Dec:add(nn.Linear(arch_dec[i],arch_dec[i+1]))
   Dec:add(nn.BatchNormalization(arch_dec[i+1]))
   Dec:add(cudnn.ReLU())
end
Dec:add(nn.Linear(arch_dec[#arch_dec], 784*2))
Dec:add(nn.View(2,28,28):setNumInputDims(1))
Dec:add(nn.Tanh())
-- AE with no noise layer
AE_ = nn.Sequential():add(Enc):add(Dec)
-- criterion training AE
criterionAE = cudnn.SpatialCrossEntropyCriterion()
criterionAE:cuda()

--[[ GAN ]]--
-- GAN generator
G = nn.Sequential()
local arch_g = convert_option(opt.archG)
table.insert(arch_g, 1, opt.noiseDim)
for i = 1, #arch_g-1 do
   G:add(nn.Linear(arch_g[i], arch_g[i+1]))
   G:add(nn.BatchNormalization(arch_g[i+1]))
   G:add(cudnn.ReLU())
end
G:add(nn.Linear(arch_g[#arch_g],opt.hidDim))
G:add(nn.Tanh())
-- GAN discriminator/critic
D = nn.Sequential()
local arch_d = convert_option(opt.archD)
D:add(nn.Linear(opt.hidDim, arch_d[1]))
D:add(nn.LeakyReLU(0.2))
for i = 1, #arch_d-1 do
   D:add(nn.Linear(arch_d[i], arch_d[i+1]))
   D:add(nn.BatchNormalization(arch_d[i+1]))
   D:add(nn.LeakyReLU(0.2))
end
D:add(nn.Linear(arch_d[#arch_d], 1))
D:add(nn.Mean())

--[[ parameter flatterning ]]--
local model = nn.Sequential():add(AE):add(D):add(G):cuda()
param_ae, gparam_ae = AE:getParameters()
ae_config = {learningRate=opt.learningRateAE, beta1=opt.beta1}
param_d, gparam_d = D:getParameters()
d_config = {learningRate=opt.learningRateD, beta1=opt.beta1}
param_g, gparam_g = G:getParameters()
g_config = {learningRate=opt.learningRateG, beta1=opt.beta1}

--[[ initialization on the models ]]--
local function initModel(model, std)
   for _, m in pairs(model:listModules()) do
      local function setWeights(module, std)
    weight = module.weight
    bias = module.bias
    if weight then weight:randn(weight:size()):mul(std) end
    if bias then bias:zero() end
      end
      setWeights(m, std)
   end
end
initModel(D, 0.02)
initModel(G, 0.02)

---------------------------------------
-------------- Train ------------------
---------------------------------------
-- init buffers
local noise  = torch.CudaTensor(opt.batchSize, opt.noiseDim)
local function make_noise()
   noise:resize(opt.batchSize, opt.noiseDim):normal()
end
local gan_grad = torch.CudaTensor(1)
local loss_real, loss_fake, loss_D, loss_fakeG, lossAE = 0, 0, 0, 0, 0
--[[training callback functions ]]--
do
   --[[ fevalAE ]]--
   function fevalAE(x)
      assert(x == param_ae)
      gparam_ae:zero()
      local output = AE:forward(x_binmnist_in)
      lossAE = criterionAE:forward(output, x_binmnist_ou)
      local derr_AE = criterionAE:backward(output, x_binmnist_ou)
      AE:backward(x_binmnist_in, derr_AE)
      return lossAE, gparam_ae
   end

   --[[ fevalD ]]--
   function fevalD(x)
      assert(x == param_d)
      gparam_d:zero()
      x:clamp(-opt.clamp, opt.clamp)
      -- on real samples
      local real = AE:get(1):forward(x_binmnist_in)
      loss_real = D:forward(real)[1]
      local dloss_real = gan_grad:fill(1)
      D:backward(real, dloss_real)
      -- on fake samples
      local fake = G:forward(noise)
      loss_fake = D:forward(fake)[1]
      local dloss_fake = gan_grad:fill(-1)
      loss_D = loss_real - loss_fake
      D:backward(fake, dloss_fake)
      return loss_d, gparam_d
   end

   --[[ fevalAE_fromGAN ]]--
   function fevalAE_fromGAN(x)
      assert(x == param_ae)
      gparam_ae:zero()
      -- on real samples
      local real = AE:get(1):forward(x_binmnist_in)
      local loss_real_ = D:forward(real)[1]
      local dloss_real = gan_grad:fill(1)
      local dreal = D:updateGradInput(real, dloss_real)
      dreal:mul(-math.abs(opt.gan2enc))
      -- fed back to the encoder
      AE:get(1):backward(x_binmnist_in, dreal)
      return loss_real, gparam_ae
   end

   --[[ fevalG ]]--
   function fevalG(x)
      assert(x == param_g)
      gparam_g:zero()

      noise:normal()
      local fake = G:forward(noise)
      loss_fakeG = D:forward(fake)[1]
      local dloss_fake = gan_grad:fill(1)
      local dG = D:updateGradInput(fake, dloss_fake)
      G:backward(noise, dG)
      return loss_fakeG, gparam_g
   end
end
-- training loop
for iEpoch = 1, opt.nEpoches do
   local tt = tt or torch.Timer()
   cutorch.synchronize()
   model:training()
   loss_real, loss_fake, loss_D, loss_fakeG, lossAE = 0, 0, 0, 0, 0
   for iIter = 1, opt.nIters do
      ------ training AE ------
      for iAE = 1, opt.nItersAE do
         prep_batch()
         optim.adam(fevalAE, param_ae, ae_config)
      end -- end for iAE = 1, 

      ------ training GAN ------
      for iGAN = 1, opt.nItersGAN do
         --- pass on D ---
         for iD = 1, opt.nItersD do
       prep_batch()
       make_noise()
       optim.adam(fevalD, param_d, d_config)
       --- backproping D into Enc ---
       optim.adam(fevalAE_fromGAN, param_ae, ae_config)
         end  -- end for iD = 1
         --- pass on G ---
         optim.adam(fevalG, param_g, g_config)
      end  -- end for iGAN = 1
   end -- end for i = 1, nIters
   model:evaluate()
   -- nan testing
   if check_nan(param_d) or check_nan(param_g) then
      error("Nan learnt.")
   end
   -- noise annealing
   AE:get(2).std = AE:get(2).std * opt.noiseAnne
   -- print message
   cutorch.synchronize()
   local tim = tt:time()['real']
   local message = string.format("epo: %d, lossD: %.4f, lossAE: %.4f, lossG: %.4f, elaps: %.2e",
                        iEpoch, -loss_D, lossAE, loss_fakeG, tim)
   print_(message)
   -- model saving and generation
   if iEpoch == opt.nEpoches then
      model:clearState()
      torch.save(string.format("%s/model.t7", opt.savename), {model, opt})
   end
   tt:reset()
end
