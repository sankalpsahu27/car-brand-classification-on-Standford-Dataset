?	"?Q*!?~@"?Q*!?~@!"?Q*!?~@      ??!       "n
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
	l?????@l?????@!l?????@      ??!       "	0?r?}@0?r?}@!0?r?}@*      ??!       2	^?pX???^?pX???!^?pX???:	??)?/@??)?/@!??)?/@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q V@?\l@y???2:?W@?"-
IteratorGetNext/_4_Recv????!z?!????!z?"c
7gradient_tape/model_1/conv1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterC??mp?!? 7??0"l
@gradient_tape/model_1/res5b_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltertuac?j?!?ne?̋?0"i
>gradient_tape/model_1/res5a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?8?9??j?!?~玩???0"l
@gradient_tape/model_1/res5c_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???X??j?!?Y????0"l
@gradient_tape/model_1/res5a_branch2b/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?_"? ?j?!?[?l<???0"i
>gradient_tape/model_1/res4a_branch1/Conv2D/Conv2DBackpropInputConv2DBackpropInput8#s`/h?!`??|(???0"m
Agradient_tape/model_1/res3b7_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????g?!6Q?{z???0"m
Agradient_tape/model_1/res3b3_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteroYv?Drg?!2?m??m??0"m
Agradient_tape/model_1/res3b6_branch2a/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~??0?dg?!zj}?+???0Q      Y@Y??A??A??a$?$??X@qE????S@y(V?_??A?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?78.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 