	"?Q*!?~@"?Q*!?~@!"?Q*!?~@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-"?Q*!?~@l?????@10?r?}@A^?pX???I??)?/@*	/?$QS@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???(???!˧?r?*J@)???(???1˧?r?*J@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?"??]???!?tK=.T@)?N?j??1Ղ??;@:Preprocessing2F
Iterator::Model??_?ǳ?!      Y@)?a?Q+L??1?-?
G?3@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI V@?\l@Q???2:?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l?????@l?????@!l?????@      ??!       "	0?r?}@0?r?}@!0?r?}@*      ??!       2	^?pX???^?pX???!^?pX???:	??)?/@??)?/@!??)?/@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q V@?\l@y???2:?W@