?	??Mb??@??Mb??@!??Mb??@      ??!       "n
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
	I?Vя@I?Vя@!I?Vя@      ??!       "	W????@}@W????@}@!W????@}@*      ??!       2	?'?y????'?y???!?'?y???:	*p?܅B@*p?܅B@!*p?܅B@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`A?i@? @y?W??W?V@?"-
IteratorGetNext/_4_Recv??X?:t?!??X?:t?"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter2??g?tp?!??$`?W??0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?B???i?!? ??Hӈ?0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?%?~?i?!d??7(J??0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQ???*?i?!??zߒ?0"m
Agradient_tape/model_1/res3b7_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?هͦh?!?|????0"i
>gradient_tape/model_1/res4a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInput_?B㏌h?!?f????0"m
Agradient_tape/model_1/res3b6_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?N?*th?!??;??0"m
Agradient_tape/model_1/res3b4_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter>h[`?g?!?=G+*??0"m
Agradient_tape/model_1/res3b1_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterNч?cg?!E? ?????0Q      Y@Y??#]?7??a`ܢ??X@qw?L????y?? ?@?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 