classifier.train(
    # input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(0, 6000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(6000, 12000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(12000, 18000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(18000, 24000), batch_size=1),
    steps=6000)

classifier.train(
    #input_fn=lambda: fn.input_functor(datanorm=testdatanorm, labelsnorm=testlabelsnorm, batch_size=1),
    input_fn=lambda: fn.input_hdf5_functor(transfer='reformed_spectra_densesapce_safe.hdf5', select=(24000, 29999), batch_size=1),
    steps=6000)

