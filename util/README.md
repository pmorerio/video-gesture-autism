### Data Format
The `train_features.h5` file is a matalb-generated HDF5 file with `DATATYPE H5T_IEEE_F32LE` and its `DATASPACE` is `SIMPLE { ( #frames, 7*7*1024 ) / ( H5S_UNLIMITED, H5S_UNLIMITED ) }` and `DATASET "features"`. Please note that the the dataset is [chunked](https://support.hdfgroup.org/HDF5/doc/Advanced/Chunking/) in order to read it faster: you may want to adjust the chunk size.

The `train_framenum.txt` file contains #frames for each video:
```
33 
35 
32 
30 
```

The `train_filename.txt` file contains the video filenames relative to the root video directory:


The `train_labels.txt`file contains the list of the video labels (autistic subject, control subject)
```
0
1
0
1
```

The same format is required for the validation and test files too.

### data_handler.py
Please note the `order='F'` in all `numpy.reshape()` calls. This is due to the fact that data files are created with Matlab which uses the Fortran indexing order. You will have to remove this parameter if that is not the case you (e.g. you create your hhdf5 dataset with h5py).
