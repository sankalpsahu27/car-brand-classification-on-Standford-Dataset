	??Mb??@??Mb??@!??Mb??@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??Mb??@I?Vя@1W????@}@A?'?y???I*p?܅B@*	?|?5HA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Qf?s@!K?}?#?X@)?Qf?s@1K?}?#?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch&qVDM???!n?	wn??)&qVDM???1n?	wn??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??5"??!咏????)k?) ?3??1ߥ?Q?~?:Preprocessing2F
Iterator::ModelgHū???!!????q??)Ǜ??,??1??B???j?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap|?_?0?s@!?2??H?X@)???QI}?1~-??b?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?7.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI`A?i@? @Q?W??W?V@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?Vя@I?Vя@!I?Vя@      ??!       "	W????@}@W????@}@!W????@}@*      ??!       2	?'?y????'?y???!?'?y???:	*p?܅B@*p?܅B@!*p?܅B@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`A?i@? @y?W??W?V@