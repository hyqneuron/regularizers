
# Things learned:
- Normalized updates lead to dead units, even when normalization happen at layer-level (normalized by per-layer 
  pre-normalization update norm).
- L2 decay for layers equipped with batchnorm are effectively a learning rate tuning device. It changes the effective
  learning rate by tuning the norm of the weight, without changing its direction.

# Ideas
- Use per-unit/layer adam (bloody hard leh)

# TODO
- Run Topographically structured decay, based on ixp2 (10)
- ixp: investigate how non-logit layer weight decays impact training
- Investigate how to combine weight decay with update normalization. Currently, regardless of what update normalization
  method I use, units get killed.

# Varying L2 decay strength on VGG13-like network
same model_conf, SGD, L2, with increasing weight_decay:
- exp1: 0.0005 -> 0.8519 (pre-terminated)
- exp2: 0.001  -> 0.8639 (50000)
- exp3: 0.002  -> 0.8816 (50000)
- exp4: 0.004  -> 0.8975 (50000)
- exp5: 0.008  -> 0.9090 (50000)
- exp6: 0.016  -> 0.9150 (49996)
- exp7: 0.032  -> 0.9101 (49993)
- exp8: 0.016  -> 0.9129 (50000)
- exp9: 0.016  -> 0.9124 (50000) (this one has momentum=0.9)

Observations:
- exp1 to exp5 all hit 100% training accuracy. exp6 did not, and exp7 was marginally worse still. This might indicate
  that the turning point for regularization is right where we lose the ability to hit 100%
- exp5, exp6 and exp7 are unstable in initial training. Test accuracy are very bad in the early epochs. Seems that only
  towards the end when the updates have very small norm does test accuracy suddenly shoot up and stabilize.

# Varying L1 decay strength on VGG13-like network, some decay in L2
same model_conf, SGDL1, L2=0.016
- axp1: 0.00016  -> 0.8734
- axp2: 0.0016   -> 0.8078
- axp3: 0.0008   -> 0.8479
- axp4: 0.00008  -> 0.8542
- axp5: 0.00032  -> 0.8717
- axp6: 0.00064  -> 0.8628

- axp4: 0.00008  -> 0.8542 (49994)
- axp1: 0.00016  -> 0.8734 (49967)
- axp5: 0.00032  -> 0.8717 (49908)
- axp6: 0.00064  -> 0.8628 (49894)
- axp3: 0.0008   -> 0.8479 (49501)
- axp2: 0.0016   -> 0.8078 (49007)

Note: it's possible that the models with big decay didn't get enough iterations as towards their final epochs the
update-to-weight ratio is still big.

Observations:
- some L1 doesn't really improve things. Maybe need further tuning.

# Varying L1 decay strength on VGG13-like network, all decay in L1
same model_conf, SGD1A
- bxp1: 0.0001  -> 0.8669 (49993)
- bxp2: 0.0002  -> 0.8671 (49970)
- bxp3: 0.0004  -> 0.8580 (49921)
- bxp4: 0.0008  -> 0.8344 (49569)

Observations:
- Replacing all L2 with L1 doesn't really improve things. Maybe need further tuning.

# variance-inverse proportional decay, with duplicate_multiplier=4
same model_conf, SGD_var_dup4, vary base_decay=weight_decay
- cxp1: 0.000005 -> 0.8449 (50000)
- cxp2: 0.00001  -> 0.8488 (50000)
- cxp3: 0.00002  -> 0.8602 (50000)
- cxp4: 0.00004  -> 0.8650 (50000)
- cxp5: 0.00008  -> 0.8813 (50000)
- cxp6: 0.00016  -> 0.8805 (50000)
- cxp7: 0.00032  -> 0.8807 (50000)
- cxp8: 0.00064  -> 0.8625 (50000)
- cxp9: 0.00128  -> 0.7610 (50000)

Observations:
- This kind of variance-inverse decay doesn't work very well, possibly because it gives very strong decay to the early
  layers, leaving the end layers with very weak decay.
- In cxp9, some early layer units are killed because of the strong decay.

# variance-inverse proportional decay, with duplicate_multiplier=2
same model_conf, SGD_var_dup2, vary base_decay=weight_decay
- dxp1: 0.00002  -> 0.8864 (50000)
- dxp2: 0.00004  -> 0.8992 (50000)
- dxp3: 0.00008  -> 0.9118 (50000)
- dxp4: 0.00016  -> 0.9131 (50000)
- dxp5: 0.00032  -> 0.9101 (50000)
- dxp6: 0.00064  -> 0.8694 (49993)

Observations:
- Among all the variance-inverse methods, this one seems to work the best.
- It keeps the weight decay relatively constant across layers. Only the final 2 layers have very weak decay because they
  are 1x1.

# [fxps.sh] variance-inverse proportional decay, with duplicate_multiplier=1
same model_conf, SGD_var_dup1, vary base_decay=weight_decay
- fxp1: 0.000005 -> 0.8834 (50000)
- fxp2: 0.00001  -> 0.8989 (50000)
- fxp3: 0.00002  -> 0.9077 (50000)
- fxp4: 0.00004  -> 0.9076 (50000)
- fxp5: 0.00008  -> 0.9086 (49997)
- fxp6: 0.00016  -> 0.8998 (49978)
- fxp7: 0.00032  -> 0.1000 (4848)
- fxp8: 0.00064  -> 0.1000 (6310)

Observations:
- This doesn't work. Variance-inverse doesn't work.

# dxp4-based, vary decay of last 2 layers
same model_conf, SGD_var_dup2_last2_mult, vary last2_mult
- gxp1: 0.1  -> 0.9131 (50000)
- dxp4: 1    -> 0.9131 (50000)
- gxp4: 3    -> 0.9130 (50000)
- gxp2: 10   -> 0.9170 (50000)
- gxp5: 30   -> 0.9151 (50000)
- gxp3: 100  -> 0.9165 (49990)

Observations:
- decay on the last 2 layers do matter. But is it on the last one only, or last two? DONE This is investigated in ixps.
  Indeed, it seems better if we only tune the decay on the last logit layer

# [hxps.sh] exp6-based, vary weight_decay from 0.010 to 0.026
same model_conf, SGD, with increasing weight_decay:
- hxp1: 0.010  -> 9108 (49998)
- hxp2: 0.012  -> 9119 (49999)
- hxp3: 0.014  -> 9163 (49999)
- hxp4: 0.016  -> 9112 (49998)
- hxp5: 0.018  -> 9088 (50000)
- hxp6: 0.020  -> 9129 (50000)
- hxp7: 0.022  -> 9082 (49999)
- hxp8: 0.024  -> 9103 (49998)
- hxp9: 0.026  -> 9092 (49999)

Observations:
- they are not nearly as good as dxp4 and several gxps

# dxp4-based, vary decay of last 1 layer
same model_conf, SGD_var_dup2_last1_mult, vary last1_mult
- ixp1: 3   -> 9147 (50000)
- ixp2: 10  -> 9188 (50000), maxed at 9202
- ixp3: 30  -> 9182 (50000)
- ixp4: 100 -> 9119 (50000)

Observations:
- Just by tuning the decay for the final layer, things can indeed improve by a bit. TODO This requires further
  validation to understand the relationship between previous-layer weight decay and final-layer weight decay. How do the
  decay in previous layers impact training, given that they are largely irrelevant under batchnorm?

# Layer-specific learning rate, based on ixp2
same model_conf, LSSGD_var_dup2_last1_mult, vary last1_mult
- jxp1: 3   -> 8672 (aborted at epoch 7)
- jxp2: 10  -> 8666 (aborted at epoch 7)
- jxp3: 30  -> 8672 (aborted at epoch 7)
- jxp4: 100 -> 8710 (aborted at epoch 7)

Observations:
- Result becomes completely insensitive to last1_mult. This might mean that previously, last1_mult mattered because it
  influenced the magnitude of gradients to the early layer
- Another thing is, it seems that stronger last1_mult > 100 seems to work better... well, needs validation

# Layer-specific learning rate, based on ixp2, with ema normalization
same model_conf, LSSGD_var_dup2_last1_mult, vary last1_mult
- kxp1: 3   -> 8728 (aborted at epoch 7)
- kxp2: 10  -> 8688 (aborted at epoch 7)
- kxp3: 100 -> 8716 (aborted at epoch 7)

Observations:
- Still quite bad. Shall we try removing the multiplication by weight norm? DONE Can do this in lxps

# Layer-specific learning rate, based on ixp2, with ema normalization, without multiplication by weight norm
same model_conf, LSSGD_var_dup2_last1_mult_nwm, vary last1_mult
- lxp1: 3   -> 8918 (50000)
- lxp2: 10  -> 8968 (50000)
- lxp3: 100 -> 8945 (50000)

Observations:
- Improvement over kxps, which have multiplication with weight norm.
- looks like any kind of normalization to update would screw things up quite a bit.
- thu.get_model_utilization(model, threshold=1e-9) shows some killed units in conv5_1 and conv5_2 (around 70). In
  comparison, ixp2, on which lxps are based, has only 4 dead units across all layers even when threshold=1e-5. This
  probably means that we need to be very careful with weight decay when we are using update normalization. TODO
  investigate how to combine weight decay with update normalization.

# Topographic weight decay, based on ixp2 (wd=0.00016, lr=0.1, last1_mult=10)
  same model_conf, trying different optimizers
- topo.a1:  SGD_topo.a  -> 9146 (50000)
- topo.b1:  SGD_topo.b  -> 9125 (50000)
- topo.ab1: SGD_topo.ab -> 9127

Observations:
- Results not statistically significant as shown by later experiments

# Topographic weight decay, based on ixp2, last1_mult=10, vary wd
same model_conf, SGD_topo.a
- topo.aa1: 0.00004  -> 9004 (50000)
- topo.aa2: 0.00008  -> 9086 (50000)
- topo.aa3: 0.00016  -> 9184 (50000)
- topo.aa4: 0.00020  -> 9127 (50000)
- topo.aa5: 0.00032  -> 9131 (49997)
- topo.aa6: 0.00064  -> 8753 (49985)

Observations:
- topo.a is capable of reaching the same accuracy as ixp2, so topo.a's result is not statistically significant yet

# Topographic weight decay, based on ixp2, wd=0.00016, vary last1_mult
same model_conf, SGD_topo.a
- topo.ad1: 2   -> 9176
- topo.ad2: 5   -> 9172
- topo.ad3: 10  -> 9141
- topo.ad4: 20  -> 9161
- topo.ad5: 40  -> 9144 (49998)
- topo.ad6: 80  -> 9154 (49997)
- topo.ad7: 160 -> 9176 (49996), hit 9191
- topo.ad8: 320 -> 9189 (49995), hit 9194
- topo.ad9: 640 -> 9160 (49996)

Observations:
- last1_mult can take on a far wider range of values than previously thought

# Topographic weight decay, based on ixp2, wd=0.00016, last1_mult=10, vary topo_base
same model_conf, SGD_topo.a
- topo.ac1: 1.0 -> 9122
- topo.ac2: 1.1 -> 9157
- topo.ac3: 1.2 -> 9173
- topo.ac4: 1.5 -> 9162
- topo.ac5: 1.8 -> 9151
- topo.ac6: 2.0 -> 9147
- topo.ac7: 2.3 -> 9176
- topo.ac8: 2.8 -> 9151
- topo.ac9: 3.0 -> 9133
- topo.ac10:4.0 -> 9150
- topo.ac11:6.0 -> 9143

Observations:
- Even in topo.ac11, I did not see the expected "grouping" effect in fc6 and conv5_2.
