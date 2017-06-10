--[[
  Main file
--]]


---------------------------------------
-------------- Setting ----------------
---------------------------------------
cmd = torch.CmdLine()
-- general setting
cmd:option('-gpuid', 1, [[device id]])
cmd:option('-save_name', '69', [[Save name]])  -- TODO
cmd:option('-ndisp', 8, [[number of display sentences]])  -- TODO
-- data setting
cmd:option('-data_file', 'SNLI/snli-15-train.hdf5', [[Path to the training *.hdf5 file from preprocess.py]])
cmd:option('-val_data_file', 'SNLI/snli-15-val.hdf5', [[Path to validation *.hdf5 file from preprocess.py]])
cmd:option('-dict_file', 'SNLI/snli-15.dict', [[Path to the dictionary from from preprocess.py]])
cmd:option('-prob_delete', 0, [[Augmentation: deletion]])
cmd:option('-prob_swap',   0, [[Augmentation: word swaping]])
cmd:option('-permute',     0, [[Augmentation: permutation]])
cmd:option('-reverse',     0, [[Augmentation: reversion]])
-- RNN settings
cmd:option('-num_layers', 1, [[Number of layers in the LSTM encoder]])
cmd:option('-rnn_size', 300, [[Size of LSTM encoder hidden states]])
cmd:option('-word_vec_size', 300, [[Word embedding sizes]])
cmd:option('-num_layers_query', 1, [[Size of LSTM decoder hidden states]])
cmd:option('-query_size', 300, [[Size of LSTM decoder hidden states]])
cmd:option('-init_dec', 0, [[1|0]])
cmd:option('-word_dropout', 0, [[1|0]])
cmd:option('-param_init', 0.1, [[Parameters of the RNN autoencoder are initialized over uniform distribution
                               with support (-param_init, param_init)]])
cmd:option('-max_grad_norm', 1, [[Gradient clipping max norm for RNN autoencoder]])
cmd:option('-teacher_forcing', 1, [[0: free running. 1: teaching forcing.
                                    2: using the greedy decode result from previous token]])
cmd:option('-lr_decay', 1, [[Learning rate decay on RNN autoencoder learning]])
cmd:option('-prealloc', 1, [[Use memory preallocation and sharing between cloned encoder/decoders]])
-- Noise injecting onto the context vector
cmd:option('-radius', 0.2, [[Standard deviation of the noise added to the code vector]])
cmd:option('-radius_anne', 0.995, [[Noise std exponential decay factor]])
-- training settings
cmd:option('-epochs', 6, [[Number of training epochs]])
cmd:option('-niters_ae', 1, [[Number of iters trained on RNN autoencoder within each iteration]])
cmd:option('-niters_gan', 1, [[Number of iters trained on WGAN within each iteration]])
cmd:option('-niters_gan_d', 5, [[Number of iterations training WGAN critic for each loop]])
cmd:option('-save_every', 50000, [[Save every this many minibatches]])
cmd:option('-print_every', 500, [[Print stats after this many minibatches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
-- GAN settings
cmd:option('-gan_z', 100, [[Size of the z vector]])
cmd:option('-arch_g', '300-300', [[Architecture of the WGAN generator G]])
cmd:option('-arch_d', '300-300', [[Architecture of the WGAN discriminator/critic D]])
cmd:option('-gan_clamp', 0.01, [[Weight clamping for the WGAN critic]])
cmd:option('-gan_toenc', -0.01, [[Multiplier to the gradient from GAN to enc]])
cmd:option('-enc_grad_norm', true, [[Regularize the gradient from different sources
                                   into the RNN encoder to have the same l2 norm]])
cmd:option('-niters_gan_schedule', '2-4-6', [[GAN training schedule: niters_gan increments at
                                            the beginning of epoch No. 2, 4 and 6 in 2-4-6 setting]])
-- optimization setting
cmd:option('-beta1', 0.9, [[beta1 for optimizers]])
cmd:option('-ae_wd', 0, [[Weight decay for RNN autoencoder training]])
cmd:option('-ae_optim', 'sgd', [[Optimizer for RNN autoencoders]])
cmd:option('-gan_gen_optim', 'adam', [[Optimizer for WGAN generator]])
cmd:option('-gan_disc_optim', 'adam', [[Optimizer for WGAN discriminator]])
cmd:option('-learning_rate_ae', 1, [[Learning rate for RNN autoencoders]])
cmd:option('-learning_rate_g', 5e-05, [[Learning rate for the WGAN generator]])
cmd:option('-learning_rate_d', 1e-05, [[Learning rate for the WGAN discriminator/critic]])
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- TODO move up
require 'nn'
require 'nngraph'
require 'cudnn'
require 'optim'
paths.dofile('data.lua')
paths.dofile('models.lua')
paths.dofile('utils.lua')

---------------------------------------
----------------- Aux -----------------
---------------------------------------
-- logging aux
os.execute("mkdir -p " .. opt.save_name)
print("Saving to " .. opt.save_name)
--[[ printing and logging aux function ]]--
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
   file = io.open(FILE_NAME, 'a')
   file:write(str .. "\n")
   file:close()
end
--[[ WGAN training schedule meta variable: GAN_SCHEDULE ]]--
GAN_SCHEDULE = {}
if opt.niters_gan_schedule:len() > 0 then
   local schedule = convert_option(opt.niters_gan_schedule)
   for k,v in pairs(schedule) do
      GAN_SCHEDULE[v] = true
   end
end
--[[ update epoch aux function ]]--
FILE_NAME = ""  -- meta variable for logging file
function update_epoch(iepo)
   FILE_NAME = string.format("%s/%d.log", opt.save_name, iepo)
   if GAN_SCHEDULE[iepo] and opt.niters_gan > 0 then
      opt.niters_gan = opt.niters_gan + 1
   end
   opt.learning_rate_ae = math.max(0.1, opt.learning_rate_ae*opt.lr_decay)
end

---------------------------------------
---------------- TRAIN ----------------
---------------------------------------
--[[ train function ]]--
function train(train_data, valid_data)
   -- initialization: variables
   local timer = torch.Timer()
   opt.train_perf = {}
   opt.val_perf = {}      
   if opt.teacher_forcing == 0 then
      word_vecs_dec.weight:zero()
   end
   -- initialization: RNN autoencoder
   param_ae, gparam_ae = modelae:getParameters()
   param_ae:uniform(-opt.param_init, opt.param_init)
   ae_config = {learningRate=opt.learning_rate_ae, beta1=opt.beta1, weightDecay=opt.ae_wd}
   -- initialization: WGAN
   local function set_weights(model, std)
      for _, m in pairs(model:listModules()) do
    if m.weight then m.weight:randn(m.weight:size()):mul(std) end
    if m.bias then m.bias:zero() end
      end  -- end for _, m 
   end -- end function set_weights
   param_d, gparam_d = gandisc:getParameters() 
   d_config = {learningRate=opt.learning_rate_d, beta1=opt.beta1}
   set_weights(gandisc, 0.02)
   param_g, gparam_g = gangen:getParameters()
   g_config = {learningRate=opt.learning_rate_g, beta1=opt.beta1}
   set_weights(gangen, 0.02)
   -- initialization: buffers, so tensors not need to be cloned
   encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.rnn_size)
   pad_proto = torch.ones(opt.max_batch_l):long()
   wordpad_proto = torch.ones(opt.max_batch_l):long():fill(3)
   -- initialization: clone encoder/decoder up to max length
   encoder_clones = clone_many_times(encoder, opt.max_sent_l)
   decoder_clones = clone_many_times(decoder, opt.max_sent_l)
   for i = 1, opt.max_sent_l do
      if encoder_clones[i].apply then
         encoder_clones[i]:apply(function(m) m:setReuse() end)
         if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
      end
      if decoder_clones[i].apply then
         decoder_clones[i]:apply(function(m) m:setReuse() end)
         if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
      end    
   end  -- end for i = 1,
   -- initialization: hidden states buffer for RNN autoencoder, fwd/bwd steps
   local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
   local h_init_query = torch.zeros(opt.max_batch_l, opt.query_size)
   if opt.gpuid >= 0 then
      h_init = h_init:cuda()
      h_init_query = h_init_query:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      pad_proto = pad_proto:cuda()
      wordpad_proto = wordpad_proto:cuda()
   end
   init_layer = {}
   for L = 1, opt.num_layers do
     table.insert(init_layer, h_init:clone())
     table.insert(init_layer, h_init:clone())
   end
   init_layer_query = {}
   for L = 1, opt.num_layers_query do
     table.insert(init_layer_query, h_init_query:clone())
     table.insert(init_layer_query, h_init_query:clone())
   end
   
   --[[ reset init_layer or init_layer_query  ]]--
   function reset_state(state, batch_l, t)
     if t == nil then
       local u = {}
       for i = 1, #state do
         state[i]:zero()
         table.insert(u, state[i][{{1, batch_l}}])
       end
       return u
     else
       local u = {[t] = {}}
       for i = 1, #state do
         state[i]:zero()
         table.insert(u[t], state[i][{{1, batch_l}}])
       end
       return u
     end
   end

   --[[ data augmentation wrapper ]]--
   function data_augment(input)
      local source, source_l, target, target_l, batch_l = unpack(input)
      if opt.prob_delete > 0 then
         source = delete(source, opt.prob_delete)
         source_l = source:size(1)
      end
      if opt.prob_swap > 0 then
         source = swap(source, opt.prob_swap)
      end
      if opt.permute == 1 then
         source = permute(source)
         target = source[{{2, source_l}}]
      end
      if opt.reverse == 1 then
        local new_target = target:clone()
        for j = 1, new_target:size(1) - 1 do
 	 new_target[j]:copy(target[target_l-j])
        end
        target = new_target
      end     
      return {source, source_l, target, target_l, batch_l}
   end

   --[[ gradient clipping ]]--
   function grad_clip(grad)
      local grad_norm = grad:norm()
      local shrinkage = opt.max_grad_norm / grad_norm
      if shrinkage < 1 then
         grad = grad:mul(shrinkage)
      end
      return grad_norm
   end

   --[[ encoder fwd pass ]]--
   function encoder_forward(input, withnoise)
      local source, source_l, target, target_l, batch_l = unpack(input)
      local rnn_state_enc = reset_state(init_layer, batch_l, 0)
      local pad = pad_proto[{{1, batch_l}}]
      local encoder_inputs = {}
      for t = 1, source_l do
         encoder_clones[t]:training()
         encoder_inputs[t] = {source[t], table.unpack(rnn_state_enc[t-1])}
         rnn_state_enc[t] = encoder_clones[t]:forward(encoder_inputs[t])
      end
      local context_input = rnn_state_enc[source_l][#rnn_state_enc[source_l]]
      local context = transferU:forward(context_input)
      -- noise injecting
      if withnoise and opt.radius > 0 then
         cnoise = cnoise or torch.CudaTensor()
         cnoise:resizeAs(context):normal(0, opt.radius)
         context:add(cnoise)
      end
      return context, encoder_inputs, rnn_state_enc
   end
 
   --[[ decoder fwd+bwd pass ]]--
   function decoder_pass(input, context, rnn_state_enc)
      local source, source_l, target, target_l, batch_l = unpack(input)
      local rnn_state_dec = reset_state(init_layer_query, batch_l, 0)
      local pad = pad_proto[{{1, batch_l}}]
      local wordpad = wordpad_proto[{{1, batch_l}}]
      local encoder_grads = encoder_grad_proto[{{1, batch_l}}]:zero()
      if opt.init_dec == 1 then
         rnn_state_dec[0][#rnn_state_dec[0]]:copy(transferL:forward(context))
      end           
      if opt.teacher_forcing == 2 then
         gd_target = gd_target or target:clone()
         gd_target:resizeAs(target):zero()
      end
      local decoder_inputs = {}
      for t = 1, target_l do
         decoder_clones[t]:training()
         if t == 1 or opt.teacher_forcing == 0 then
            decoder_inputs[t] = {pad, table.unpack(rnn_state_dec[t-1])}
         elseif opt.teacher_forcing == 2 then  -- gd
            decoder_inputs[t] = {gd_target[t-1], table.unpack(rnn_state_dec[t-1])}
         else
            if torch.uniform() < opt.word_dropout then
          decoder_inputs[t] = {wordpad, table.unpack(rnn_state_dec[t-1])}
            else
          decoder_inputs[t] = {target[t-1], table.unpack(rnn_state_dec[t-1])}
            end	  
         end	
         rnn_state_dec[t] = decoder_clones[t]:forward(decoder_inputs[t])
         if opt.teacher_forcing == 2 then
            local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
            local pred = wordgen:forward(pred_input)
            local _, pred_argmax = pred:max(2)
            gd_target[t]:copy(pred_argmax)
         end
      end  -- end for t = 1
 
      local drnn_state_dec = reset_state(init_layer_query, batch_l)
      for t = target_l, 1, -1 do
         local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
         local pred = wordgen:forward(pred_input)
         train_loss = train_loss + criterion:forward(pred, target[t])  -- inventory
         local _, pred_argmax = pred:max(2)
         train_correct = train_correct + pred_argmax:cuda():eq(target[t]):sum()  -- inventory
         local dl_dpred = criterion:backward(pred, target[t])
         dl_dpred:div(batch_l)	
         local generator_grads = wordgen:backward(pred_input, dl_dpred)
         drnn_state_dec[#drnn_state_dec]:add(generator_grads[1])
         encoder_grads:add(generator_grads[2])
         local dl_ddec = decoder_clones[t]:backward(decoder_inputs[t-1], drnn_state_dec)
         for j = 1, #drnn_state_dec do
            drnn_state_dec[j]:copy(dl_ddec[j+1])
         end
      end  -- end for t = target_l
 
      local dcontext
      if opt.init_dec == 1 then
         local dcontextL = transferL:backward(context, drnn_state_dec[#drnn_state_dec])
         encoder_grads:add(dcontextL)
      end
      local context_input = rnn_state_enc[source_l][#rnn_state_enc[source_l]]
      dcontext = transferU:backward(context_input, encoder_grads)   
      return dcontext, rnn_state_dec
   end

   --[[ encoder bwd pass ]]--
   function encoder_backward(input, encoder_inputs, dcontext)
      local source, source_l, target, target_l, batch_l = unpack(input)
      local drnn_state_enc = reset_state(init_layer, batch_l)
      drnn_state_enc[#drnn_state_enc]:copy(dcontext)
      for t = source_l, 1, -1 do
         local dl_denc = encoder_clones[t]:backward(encoder_inputs[t], drnn_state_enc)
         for j = 1, #drnn_state_enc do
            drnn_state_enc[j]:copy(dl_denc[j+1])
         end
      end 
   end
   
   --[[ learning rate decay and model saving ]]--
   function decay_lr_and_save(id)
 	   local save_name = string.format('%s/model_%d.t7', opt.save_name, id)
      print_('saving checkpoint to ' .. save_name, true)
      modelgan:clearState()
      torch.save(save_name, { {modelae, param_ae, gparam_ae},
                              {modelgan, param_g, gparam_g, param_d, gparam_d},
                            opt})
   end
 
   --[[ train loop function for one epoch ]]--
   function train_batch(data, epoch)
     -- inventory variables
     train_loss = 0
     train_correct = 0
     start_time = timer:time().real
     num_words_target = 0
     num_words_source = 0
     dcontext_norm_ae, dcontext_norm_gan = 0, 0
     -- shuffle minibatch order
     local batch_order = torch.randperm(data.length)
     -- main loop
     for iter = 1, data:size() do

       -------------------------------
       ---- RNN autoencoder phase ----
       -------------------------------
       modelae:training()
       local d = data[batch_order[iter]]
       local input = data_augment(d)
       local function fevalAE(x)
          assert(x == param_ae)
          gparam_ae:zero()
          -- fwd/bwd on encoder/decoder
          local context, encoder_inputs, rnn_state_enc = encoder_forward(input, true)
          local dcontext, rnn_state_dec = decoder_pass(input, context, rnn_state_enc)
          dcontext_norm_ae = dcontext:norm()
          encoder_backward(input, encoder_inputs, dcontext)
          -- bookkeeping
          local source, source_l, target, target_l, batch_l = unpack(input)
          num_words_target = num_words_target + batch_l*target_l
          num_words_source = num_words_source + batch_l*source_l
          -- gradient handling
          if opt.teacher_forcing == 0 then
             word_vecs_dec.gradWeight:zero()
          end      
          grad_norm_ae = grad_clip(gparam_ae)  -- clipping
          return errAE, gparam_ae
       end  -- end local function fevalAE(x)
       if opt.ae_optim == "adam" then
          optim.adam(fevalAE, param_ae, ae_config)
       elseif opt.ae_optim == "sgd" then
          optim.sgd(fevalAE, param_ae, ae_config)
       end

       --------------------
       ---- GAN phase -----
       --------------------
       modelgan:training()
       for igan = 1, opt.niters_gan do
          --- WGAN discriminator/critic pass ---
          for igan_d = 1, opt.niters_gan_d do
             -- feed a seen sample within this epoch -- good for early training
             local d = data[batch_order[torch.random(iter)]] 
             local input = data_augment(d)
             noise_z:resize(input[1]:size(2), opt.gan_z):normal()
             -- real sample code: encoder fwd
             local real, encoder_inputs, rnn_state_enc = encoder_forward(input, false)
             local function fevalD(x)
               assert( x == param_d )
               gparam_d:zero()
               x:clamp(-opt.gan_clamp, opt.gan_clamp)
               -- real sample
               err_real = gandisc:forward(real)[1]
               local derr_real = gan_grad:fill(1)
               gandisc:backward(real, derr_real)
               -- fake sample
               local fake = gangen:forward(noise_z)
               err_fake = gandisc:forward(fake)[1]
               local derr_fake = gan_grad:fill(-1)
               gandisc:backward(fake, derr_fake)
               return err_real-err_fake, gparam_d
             end  -- end local function fevalD
             if opt.gan_disc_optim == "adam" then
                optim.adam(fevalD, param_d, d_config)
             elseif opt.gan_disc_optim == "sgd" then
                optim.sgd(fevalD, param_d, d_config)
             elseif opt.gan_disc_optim == "rmsprop" then
                optim.rmsprop(fevalD, param_d, d_config)	 
             end
             
             ---- GAN -> RNN encoder ----
             local function fevalAE_fromGAN(x)
                assert( x == param_ae )
                gparam_ae:zero()
                -- GAN discriminator pass
                local err_real_ = gandisc:forward(real)[1]
                local derr_real = gan_grad:fill(1)
                local derr_context = gandisc:updateGradInput(real, derr_real)
                local context_input = rnn_state_enc[input[2]][#rnn_state_enc[input[2]]]
                local dcontext = transferU:backward(context_input, derr_context)
                -- gradient norm regularize to be same
                dcontext_norm_gan = dcontext:norm()
                if opt.enc_grad_norm then
                   local ratio = dcontext_norm_gan / dcontext_norm_ae
                   if ratio > 0 then dcontext:div(ratio) end
                end  -- end if opt.enc_grad_norm
                dcontext:mul(-math.abs(opt.gan_toenc))
                -- encoder bwd pass
                encoder_backward(input, encoder_inputs, dcontext)
                grad_norm_from_gan = grad_clip(gparam_ae)
                return err_real_, gparam_ae
             end  -- end local function fevalAE_fromGAN
             if opt.ae_optim == "adam" then
                optim.adam(fevalAE_fromGAN, param_ae, ae_config)
             elseif opt.ae_optim == "sgd" then
                optim.sgd(fevalAE_fromGAN, param_ae, ae_config)
             end
          end  -- end for igan_d
 
          --- WGAN generator pass ---
          noise_z:resize(data[1][1]:size(2), opt.gan_z):normal()
          local function fevalG(x)
             assert( x == param_g )
             gparam_g:zero()
             -- fwd/bwd
             local fake = gangen:forward(noise_z)
             errG_fake = gandisc:forward(fake)[1]
             local derr_fake = gan_grad:fill(1)
             local derr_fake = gandisc:updateGradInput(fake, derr_fake)
             gangen:backward(noise_z, derr_fake)
             return errG_fake, gparam_g
          end  -- end local function fevalG
          if opt.gan_gen_optim == "adam" then
             optim.adam(fevalG, param_g, g_config)
          elseif opt.gan_gen_optim == "sgd" then
             optim.sgd(fevalG, param_g, g_config)
          elseif opt.gan_gen_optim == "rmsprop" then
             optim.rmsprop(fevalG, param_g, g_config)	   
          end
       end  -- end for igan = 1

       --------------------
       ----- logging ------
       --------------------
       modelgan:evaluate()
       modelae:evaluate()
       -- frequent print/log WGAN related stats
       if opt.niters_gan>0 and iter%100==0 then
          print_(('errD_real: %.4e, errD_fake: %.4e, errD: %.4e, errG_fake: %.4e'):format(
               err_real, err_fake, -(err_real-err_fake), errG_fake))
       end
       -- print/log generated text, with the nearest neighbor retrieval on the valid set
       if opt.niters_gan>0 and iter%1000==0 then
          -- run eval: to get the code_cache for nearest neighbor retrieval
          local _ = eval(valid_data)
          print(_)  -- TODO
          -- generate text
          gentext(opt.ndisp)
       end
       -- non-frequent print/log comprehensive stats
       local time_taken = timer:time().real - start_time
       if iter % opt.print_every == 0 then
          print_(('Epoch: %d, Batch: %d,  PPL: %.2f, Acc: %.2f, '
               .. '|Param|: %.2f, |GParamAE|: %.4e, |GParamF|: %.4e, '
               .. 'Rs: %.4e, nGAN: %d, LR: %.4f, Elap: %d tokens/sec'):format(
               epoch, iter, math.exp(train_loss/num_words_target),
               train_correct/num_words_target, param_ae:norm(),
               grad_norm_ae, grad_norm_from_gan or 0, opt.radius,
               opt.niters_gan, opt.learning_rate_ae, num_words_target/time_taken))
       end
       -- noise radius annealing, per 100 mini-batches
       opt.radius = iter%100==0 and opt.radius*opt.radius_anne or opt.radius
     end -- end for iter = 1, data:size()
     return train_correct, num_words_target
   end  -- end funcion train_batch

   ----- overall training loop -----
   for epoch = 1, opt.epochs do
      -- update meta variable
      update_epoch(epoch)
      print("logging into " .. FILE_NAME)
      -- entering this epoch
      local total_correct, total_nonzeros = train_batch(train_data, epoch)
      opt.train_perf[#opt.train_perf+1] = total_correct/total_nonzeros
      opt.val_perf[#opt.val_perf+1] = eval(valid_data)
      -- epoch ending print
      print_('Train Accuracies')
      print_(opt.train_perf)
      print_('Valid Accuracies')
      print_(opt.val_perf)
      -- checkpoint
      decay_lr_and_save(epoch)
      collectgarbage()
   end  -- end for epoch = 1,
end  -- end function train

--[[ evaluate function ]]--
function eval(data)
   local loss = 0
   local total = 0
   local correct = 0
   -- for nearest neighbor retrieval: caching the codes
   code_cache = code_cache or torch.CudaTensor()
   -- initialize code_cache with the data size
   local numeval = 0
   for i = 1, data:size() do
      numeval = numeval + data[i][1]:size(2)
   end
   code_cache:resize(numeval, opt.rnn_size):zero()
   -- iterate through the data
   local id = 1
   for i = 1, data:size() do    
     local d = data[i]
     local input = data_augment(d)
     local source, source_l, target, target_l, batch_l = unpack(input)
     local rnn_state_enc = reset_state(init_layer, batch_l, 0)
     local pad = pad_proto[{{1, batch_l}}]
     local encoder_inputs = {}
     -- encoder fwd
     for t = 1, source_l do
         encoder_clones[t]:evaluate()
         encoder_inputs[t] = {source[t], table.unpack(rnn_state_enc[t-1])}
         rnn_state_enc[t] = encoder_clones[t]:forward(encoder_inputs[t])
     end  -- end for t = 1, 
     local context_input = rnn_state_enc[source_l][#rnn_state_enc[source_l]]
     local context = transferU:forward(context_input)
     -- caching the code into code_cache
     code_cache:narrow(1, id, source:size(2)):copy(context)
     -- decoder fwd
     local rnn_state_dec = reset_state(init_layer_query, batch_l, 0)
     if opt.init_dec == 1 then
        rnn_state_dec[0][#rnn_state_dec[0]]:copy(transferL:forward(context))
     end
     local decoder_inputs = {}
     local pred_argmax
     for t = 1, target_l do
        decoder_clones[t]:evaluate()
        if t == 1 or opt.teacher_forcing == 0 then
           decoder_inputs[t] = {pad, table.unpack(rnn_state_dec[t-1])}
        else
           decoder_inputs[t] = {pred_argmax[{{}, 1}], table.unpack(rnn_state_dec[t-1])}
        end	 -- end if t == 1 or
        rnn_state_dec[t] = decoder_clones[t]:forward(decoder_inputs[t])
        local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
        local pred = wordgen:forward(pred_input)
        loss = loss + criterion:forward(pred, target[t])
        __, pred_argmax = pred:max(2)
        correct = correct + pred_argmax:cuda():eq(target[t]):sum()      
     end  -- end for t = 1, target_l
     total = total + batch_l * target_l
     id = id + source:size(2)
   end -- end for i = 1, data:size()   
   collectgarbage()
   return correct/total
end

--[[ display generated text ]]--
function gentext(n)
   -- set the generated sent length
   local genlength = opt.max_sent_l - 2  -- not counting <s> and </s>
   local pad = pad_proto[{{1, n}}]
   -- load dictionary
   if idict==nil then
      _, idict = load_dict(opt.dict_file)
   end
   -- use a fixed vector for visualization
   assert(n == opt.ndisp)
   noise_v = noise_v or torch.CudaTensor(n, opt.gan_z):normal()
   -- buffer initialization
   -- + genexpo: batch of generated sentences in the tensor format
   -- + gennear: batch of retrieved sentences in the tensor format
   -- + near_context: used for retrival
   genexpo = genexpo or torch.LongTensor(genlength, n):zero()
   gennear = gennear or torch.LongTensor(genlength, n):zero()
   near_context = near_context or torch.CudaTensor()
   dist_context = dist_context or torch.CudaTensor()
   -- WGAN generator fwd
   local fake_context = gangen:forward(noise_v)
   near_context:resizeAs(fake_context)
   --[[ generate samples inline function ]]--
   local function generate_sample(context, result)
      local rnn_state_dec = reset_state(init_layer_query, n, 0) 
      if opt.init_dec == 1 then
         rnn_state_dec[0][#rnn_state_dec[0]]:copy(transferL:forward(context))
      end     
      local decoder_inputs = {}
      local pred_argmax
      -- decoder fwd
      for t = 1, genlength do
         decoder_clones[t]:evaluate()
         if t == 1 or opt.teacher_forcing == 0 then
            decoder_inputs[t] = {pad, table.unpack(rnn_state_dec[t-1])}
         else
            decoder_inputs[t] = {pred_argmax[{{}, 1}], table.unpack(rnn_state_dec[t-1])}
         end  -- end if t == 1 or 
         rnn_state_dec[t] = decoder_clones[t]:forward(decoder_inputs[t])
         local pred_input = {rnn_state_dec[t][#rnn_state_dec[t]], context}
         local pred = wordgen:forward(pred_input)
         _, pred_argmax = pred:max(2)
         -- copy into result tensor
         result:select(1,t):copy(pred_argmax)
      end  -- ned for t = 1, genlength
      return result
   end  -- end local function generate_sample
   -- run the generate_sample function on the generated code
   genexpo = generate_sample(fake_context, genexpo)
   -- nearest neighbor retrieval based on the code space
   dist_context:resizeAs(code_cache):copy(code_cache)
   local dist_expand = dist_context:view(1,dist_context:size(1),dist_context:size(2))
                                   :repeatTensor(n,1,1)
   local fake_expand = fake_context:view(n,1,fake_context:size(2))
                                   :expand(n,dist_context:size(1),fake_context:size(2))
   local dist_expand = dist_expand:add(-fake_expand):pow(2):mean(3):squeeze(3)
   local _, imax = dist_expand:min(2)
   imax = imax:squeeze(2)
   for j = 1, imax:size(1) do
      near_context[j]:copy(code_cache[imax[j]])
   end
   -- run the generate_sample function on the retrieved code
   gennear = generate_sample(near_context, gennear)
   -- printing and logging
   for i = 1, n do
      -- display genexpo
      local this_sample = genexpo:select(2,i)
      local this_str = convert_to_word(this_sample, idict)
      print_(string.format("[%d] %s", i, this_str))
      -- display gennear
      local this_sample = gennear:select(2,i)
      local this_str = convert_to_word(this_sample, idict)
      print_(string.format("[%d] %s", i, this_str))
      print_(" ")
   end  -- end for i = 1, n
   collectgarbage()
end  -- end function gentext

---------------------------------------
---------------- MAIN -----------------
---------------------------------------
function main()
   -- TODO take out
   if opt.gpuid >= 0 then
     print('using CUDA on GPU ' .. opt.gpuid .. '...')
     require 'cutorch'
     require 'cunn'
     cutorch.setDevice(opt.gpuid)
     cutorch.manualSeed(opt.seed)
   end
   ----------------------
   ------- data ---------
   ----------------------
   train_data = data.new(opt, opt.data_file)
   valid_data = data.new(opt, opt.val_data_file)
   opt.max_sent_l = train_data.source:size(2)
   opt.max_batch_l = train_data.batch_l:max()    
   print(string.format('Vocab size: %d', train_data.vocab_size))
   print(string.format('Max sent len: %d', opt.max_sent_l))
   ----------------------
   ------- model --------
   ----------------------
   preallocateMemory(opt.prealloc)  -- memory optimization
   ------ RNN encoder-decoder ------
   encoder = make_lstm(opt.word_vec_size, opt.rnn_size, train_data.vocab_size,
                       opt.num_layers, opt, 'enc')
   if opt.teacher_forcing == 1 or opt.teacher_forcing == 2 then
      decoder = make_lstm(opt.query_size, opt.query_size, train_data.vocab_size,
                          opt.num_layers_query, opt, 'dec')
   else
      decoder = make_lstm(opt.query_size, opt.query_size, 1,
                          opt.num_layers_query, opt, 'dec')
   end
   ------ word generator ------
   wordgen = make_word_generator(opt.rnn_size, opt.query_size, train_data.vocab_size)
   criterion = nn.ClassNLLCriterion()
   criterion.sizeAverage = false
   ------ transition modules, from encoder to decoder ------
   -- transferU: maps the code vector (context) onto a Unit ball,
   --            i.e., having norm=1.
   transferU = nn.Normalize(2)
   -- transferL: a Linear module maps the code vector to initialize first decoder state,
   --            only being used when opt.init_dec == 1
   transferL = nn.Linear(opt.rnn_size, opt.query_size, false)
   ------ WGAN ------
   gangen = make_gan_generator(opt.arch_g, opt.gan_z, opt.rnn_size)
   gandisc = make_gan_discriminator(opt.arch_d, opt.query_size)
   -- WGAN related buffers
   gan_grad = torch.CudaTensor(1)
   noise_z = torch.CudaTensor()
   ----------------------
   ----- train prep -----
   ----------------------
   layers = {encoder, decoder, wordgen}
   table.insert(layers, transferU)  
   if opt.init_dec == 1 then
      table.insert(layers, transferL)
   end
   -- index the embedding layers for RNN encoder-decoder
   local function get_layer(layer)
      if layer.name == 'word_vecs_enc' then
         word_vecs_enc = layer
      elseif layer.name == 'word_vecs_dec' then
         word_vecs_dec = layer
      end  -- end if layers.name  
   end  -- end local function get_layer
   encoder:apply(get_layer)
   decoder:apply(get_layer)
   -- CUDAize all models
   for i = 1, #layers do
      layers[i]:cuda()
   end
   gangen:cuda()
   gandisc:cuda()
   criterion:cuda()
   -- containers for every part
   modelgan = nn.Sequential():add(gangen):add(gandisc)
   modelae = nn.Sequential()
   for _,v in pairs(layers) do
      modelae:add(v)
   end
   train(train_data, valid_data)
end

main()
