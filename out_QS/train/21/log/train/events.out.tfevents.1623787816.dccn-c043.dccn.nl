       ?K"	   JB2?Abrain.Event:2??`1?     ????	 #JB2?A"??
?
placeholders/signalsPlaceholder*
dtype0*%
shape:??????????<*0
_output_shapes
:??????????<
n
placeholders/labelsPlaceholder*
shape:?????????*#
_output_shapes
:?????????*
dtype0
Y
placeholders/is_trainingPlaceholder*
dtype0
*
shape: *
_output_shapes
: 
t
placeholders/loss_weightsPlaceholder*#
_output_shapes
:?????????*
shape:?????????*
dtype0
s
placeholders/seq_lengthsPlaceholder*
dtype0*
shape:?????????*#
_output_shapes
:?????????
[
global_step/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
o
global_step
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 
?
global_step/AssignAssignglobal_stepglobal_step/initial_value*
_output_shapes
: *
_class
loc:@global_step*
use_locking(*
validate_shape(*
T0
j
global_step/readIdentityglobal_step*
_output_shapes
: *
_class
loc:@global_step*
T0
\
global_epoch/initial_valueConst*
dtype0*
value	B : *
_output_shapes
: 
p
global_epoch
VariableV2*
shape: *
dtype0*
shared_name *
	container *
_output_shapes
: 
?
global_epoch/AssignAssignglobal_epochglobal_epoch/initial_value*
_class
loc:@global_epoch*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
m
global_epoch/readIdentityglobal_epoch*
T0*
_output_shapes
: *
_class
loc:@global_epoch
?
=cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/shapeConst*%
valueB"?         @   *-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
_output_shapes
:*
dtype0
?
<cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
_output_shapes
: *
dtype0
?
>cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/stddevConst*-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
dtype0*
_output_shapes
: *
valueB
 *???=
?
Gcnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal=cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/shape*'
_output_shapes
:?@*
seed2 *
T0*

seed *-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
dtype0
?
;cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/mulMulGcnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal>cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/stddev*-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*'
_output_shapes
:?@*
T0
?
7cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normalAdd;cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/mul<cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal/mean*
T0*'
_output_shapes
:?@*-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel
?
cnn/conv1d_1/conv2d/kernel
VariableV2*
	container *-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
dtype0*'
_output_shapes
:?@*
shared_name *
shape:?@
?
!cnn/conv1d_1/conv2d/kernel/AssignAssigncnn/conv1d_1/conv2d/kernel7cnn/conv1d_1/conv2d/kernel/Initializer/truncated_normal*
validate_shape(*-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel*
T0*'
_output_shapes
:?@*
use_locking(
?
cnn/conv1d_1/conv2d/kernel/readIdentitycnn/conv1d_1/conv2d/kernel*
T0*'
_output_shapes
:?@*-
_class#
!loc:@cnn/conv1d_1/conv2d/kernel
r
!cnn/conv1d_1/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
cnn/conv1d_1/conv2d/Conv2DConv2Dplaceholders/signalscnn/conv1d_1/conv2d/kernel/read*0
_output_shapes
:??????????@*
	dilations
*
data_formatNHWC*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
3cnn/bn_1/batch_normalization/gamma/Initializer/onesConst*
dtype0*
valueB@*  ??*
_output_shapes
:@*5
_class+
)'loc:@cnn/bn_1/batch_normalization/gamma
?
"cnn/bn_1/batch_normalization/gamma
VariableV2*
dtype0*5
_class+
)'loc:@cnn/bn_1/batch_normalization/gamma*
shared_name *
	container *
shape:@*
_output_shapes
:@
?
)cnn/bn_1/batch_normalization/gamma/AssignAssign"cnn/bn_1/batch_normalization/gamma3cnn/bn_1/batch_normalization/gamma/Initializer/ones*
use_locking(*5
_class+
)'loc:@cnn/bn_1/batch_normalization/gamma*
_output_shapes
:@*
validate_shape(*
T0
?
'cnn/bn_1/batch_normalization/gamma/readIdentity"cnn/bn_1/batch_normalization/gamma*5
_class+
)'loc:@cnn/bn_1/batch_normalization/gamma*
T0*
_output_shapes
:@
?
3cnn/bn_1/batch_normalization/beta/Initializer/zerosConst*4
_class*
(&loc:@cnn/bn_1/batch_normalization/beta*
_output_shapes
:@*
valueB@*    *
dtype0
?
!cnn/bn_1/batch_normalization/beta
VariableV2*
shape:@*
shared_name *4
_class*
(&loc:@cnn/bn_1/batch_normalization/beta*
_output_shapes
:@*
	container *
dtype0
?
(cnn/bn_1/batch_normalization/beta/AssignAssign!cnn/bn_1/batch_normalization/beta3cnn/bn_1/batch_normalization/beta/Initializer/zeros*
_output_shapes
:@*4
_class*
(&loc:@cnn/bn_1/batch_normalization/beta*
validate_shape(*
use_locking(*
T0
?
&cnn/bn_1/batch_normalization/beta/readIdentity!cnn/bn_1/batch_normalization/beta*
_output_shapes
:@*
T0*4
_class*
(&loc:@cnn/bn_1/batch_normalization/beta
?
:cnn/bn_1/batch_normalization/moving_mean/Initializer/zerosConst*;
_class1
/-loc:@cnn/bn_1/batch_normalization/moving_mean*
dtype0*
_output_shapes
:@*
valueB@*    
?
(cnn/bn_1/batch_normalization/moving_mean
VariableV2*
shape:@*;
_class1
/-loc:@cnn/bn_1/batch_normalization/moving_mean*
dtype0*
	container *
_output_shapes
:@*
shared_name 
?
/cnn/bn_1/batch_normalization/moving_mean/AssignAssign(cnn/bn_1/batch_normalization/moving_mean:cnn/bn_1/batch_normalization/moving_mean/Initializer/zeros*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0*;
_class1
/-loc:@cnn/bn_1/batch_normalization/moving_mean
?
-cnn/bn_1/batch_normalization/moving_mean/readIdentity(cnn/bn_1/batch_normalization/moving_mean*;
_class1
/-loc:@cnn/bn_1/batch_normalization/moving_mean*
T0*
_output_shapes
:@
?
=cnn/bn_1/batch_normalization/moving_variance/Initializer/onesConst*
_output_shapes
:@*
dtype0*
valueB@*  ??*?
_class5
31loc:@cnn/bn_1/batch_normalization/moving_variance
?
,cnn/bn_1/batch_normalization/moving_variance
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *?
_class5
31loc:@cnn/bn_1/batch_normalization/moving_variance
?
3cnn/bn_1/batch_normalization/moving_variance/AssignAssign,cnn/bn_1/batch_normalization/moving_variance=cnn/bn_1/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
_output_shapes
:@*?
_class5
31loc:@cnn/bn_1/batch_normalization/moving_variance*
T0*
validate_shape(
?
1cnn/bn_1/batch_normalization/moving_variance/readIdentity,cnn/bn_1/batch_normalization/moving_variance*
_output_shapes
:@*?
_class5
31loc:@cnn/bn_1/batch_normalization/moving_variance*
T0
?
+cnn/bn_1/batch_normalization/FusedBatchNormFusedBatchNormcnn/conv1d_1/conv2d/Conv2D'cnn/bn_1/batch_normalization/gamma/read&cnn/bn_1/batch_normalization/beta/read-cnn/bn_1/batch_normalization/moving_mean/read1cnn/bn_1/batch_normalization/moving_variance/read*
epsilon%o?:*
is_training( *
data_formatNHWC*
T0*H
_output_shapes6
4:??????????@:@:@:@:@
g
"cnn/bn_1/batch_normalization/ConstConst*
dtype0*
valueB
 *?p}?*
_output_shapes
: 
z

cnn/relu_1Relu+cnn/bn_1/batch_normalization/FusedBatchNorm*0
_output_shapes
:??????????@*
T0
?
%cnn/maxpool1d_1/max_pooling2d/MaxPoolMaxPool
cnn/relu_1*/
_output_shapes
:?????????<@*
strides
*
data_formatNHWC*
T0*
ksize
*
paddingSAME
w
cnn/drop_1/cond/SwitchSwitchplaceholders/is_trainingplaceholders/is_training*
T0
*
_output_shapes
: : 
_
cnn/drop_1/cond/switch_tIdentitycnn/drop_1/cond/Switch:1*
T0
*
_output_shapes
: 
]
cnn/drop_1/cond/switch_fIdentitycnn/drop_1/cond/Switch*
_output_shapes
: *
T0

^
cnn/drop_1/cond/pred_idIdentityplaceholders/is_training*
T0
*
_output_shapes
: 
|
cnn/drop_1/cond/dropout/rateConst^cnn/drop_1/cond/switch_t*
valueB
 *   ?*
_output_shapes
: *
dtype0
?
cnn/drop_1/cond/dropout/ShapeShape&cnn/drop_1/cond/dropout/Shape/Switch:1*
out_type0*
T0*
_output_shapes
:
?
$cnn/drop_1/cond/dropout/Shape/SwitchSwitch%cnn/maxpool1d_1/max_pooling2d/MaxPoolcnn/drop_1/cond/pred_id*
T0*J
_output_shapes8
6:?????????<@:?????????<@*8
_class.
,*loc:@cnn/maxpool1d_1/max_pooling2d/MaxPool
}
cnn/drop_1/cond/dropout/sub/xConst^cnn/drop_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
cnn/drop_1/cond/dropout/subSubcnn/drop_1/cond/dropout/sub/xcnn/drop_1/cond/dropout/rate*
_output_shapes
: *
T0
?
*cnn/drop_1/cond/dropout/random_uniform/minConst^cnn/drop_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
?
*cnn/drop_1/cond/dropout/random_uniform/maxConst^cnn/drop_1/cond/switch_t*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
4cnn/drop_1/cond/dropout/random_uniform/RandomUniformRandomUniformcnn/drop_1/cond/dropout/Shape*/
_output_shapes
:?????????<@*

seed *
seed2 *
dtype0*
T0
?
*cnn/drop_1/cond/dropout/random_uniform/subSub*cnn/drop_1/cond/dropout/random_uniform/max*cnn/drop_1/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
?
*cnn/drop_1/cond/dropout/random_uniform/mulMul4cnn/drop_1/cond/dropout/random_uniform/RandomUniform*cnn/drop_1/cond/dropout/random_uniform/sub*/
_output_shapes
:?????????<@*
T0
?
&cnn/drop_1/cond/dropout/random_uniformAdd*cnn/drop_1/cond/dropout/random_uniform/mul*cnn/drop_1/cond/dropout/random_uniform/min*/
_output_shapes
:?????????<@*
T0
?
cnn/drop_1/cond/dropout/addAddcnn/drop_1/cond/dropout/sub&cnn/drop_1/cond/dropout/random_uniform*/
_output_shapes
:?????????<@*
T0
}
cnn/drop_1/cond/dropout/FloorFloorcnn/drop_1/cond/dropout/add*/
_output_shapes
:?????????<@*
T0
?
cnn/drop_1/cond/dropout/truedivRealDiv&cnn/drop_1/cond/dropout/Shape/Switch:1cnn/drop_1/cond/dropout/sub*
T0*/
_output_shapes
:?????????<@
?
cnn/drop_1/cond/dropout/mulMulcnn/drop_1/cond/dropout/truedivcnn/drop_1/cond/dropout/Floor*/
_output_shapes
:?????????<@*
T0

cnn/drop_1/cond/IdentityIdentitycnn/drop_1/cond/Identity/Switch*
T0*/
_output_shapes
:?????????<@
?
cnn/drop_1/cond/Identity/SwitchSwitch%cnn/maxpool1d_1/max_pooling2d/MaxPoolcnn/drop_1/cond/pred_id*8
_class.
,*loc:@cnn/maxpool1d_1/max_pooling2d/MaxPool*
T0*J
_output_shapes8
6:?????????<@:?????????<@
?
cnn/drop_1/cond/MergeMergecnn/drop_1/cond/Identitycnn/drop_1/cond/dropout/mul*1
_output_shapes
:?????????<@: *
N*
T0
?
?cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   ?   */
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel
?
>cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
?
@cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *??M=*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*
dtype0*
_output_shapes
: 
?
Icnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/shape*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*
dtype0*
T0*'
_output_shapes
:@?*

seed *
seed2 
?
=cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/mulMulIcnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal@cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*'
_output_shapes
:@?
?
9cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normalAdd=cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/mul>cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal/mean*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*
T0*'
_output_shapes
:@?
?
cnn/conv1d_2_1/conv2d/kernel
VariableV2*
shape:@?*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*'
_output_shapes
:@?*
	container *
shared_name *
dtype0
?
#cnn/conv1d_2_1/conv2d/kernel/AssignAssigncnn/conv1d_2_1/conv2d/kernel9cnn/conv1d_2_1/conv2d/kernel/Initializer/truncated_normal*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel*'
_output_shapes
:@?*
use_locking(*
T0*
validate_shape(
?
!cnn/conv1d_2_1/conv2d/kernel/readIdentitycnn/conv1d_2_1/conv2d/kernel*
T0*'
_output_shapes
:@?*/
_class%
#!loc:@cnn/conv1d_2_1/conv2d/kernel
t
#cnn/conv1d_2_1/conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
cnn/conv1d_2_1/conv2d/Conv2DConv2Dcnn/drop_1/cond/Merge!cnn/conv1d_2_1/conv2d/kernel/read*
	dilations
*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
paddingSAME*0
_output_shapes
:?????????<?*
strides

?
5cnn/bn_2_1/batch_normalization/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*7
_class-
+)loc:@cnn/bn_2_1/batch_normalization/gamma*
valueB?*  ??
?
$cnn/bn_2_1/batch_normalization/gamma
VariableV2*
dtype0*
	container *
shared_name *
_output_shapes	
:?*
shape:?*7
_class-
+)loc:@cnn/bn_2_1/batch_normalization/gamma
?
+cnn/bn_2_1/batch_normalization/gamma/AssignAssign$cnn/bn_2_1/batch_normalization/gamma5cnn/bn_2_1/batch_normalization/gamma/Initializer/ones*
T0*
_output_shapes	
:?*7
_class-
+)loc:@cnn/bn_2_1/batch_normalization/gamma*
validate_shape(*
use_locking(
?
)cnn/bn_2_1/batch_normalization/gamma/readIdentity$cnn/bn_2_1/batch_normalization/gamma*
_output_shapes	
:?*
T0*7
_class-
+)loc:@cnn/bn_2_1/batch_normalization/gamma
?
5cnn/bn_2_1/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:?*6
_class,
*(loc:@cnn/bn_2_1/batch_normalization/beta*
dtype0*
valueB?*    
?
#cnn/bn_2_1/batch_normalization/beta
VariableV2*
dtype0*
shape:?*
shared_name *6
_class,
*(loc:@cnn/bn_2_1/batch_normalization/beta*
_output_shapes	
:?*
	container 
?
*cnn/bn_2_1/batch_normalization/beta/AssignAssign#cnn/bn_2_1/batch_normalization/beta5cnn/bn_2_1/batch_normalization/beta/Initializer/zeros*6
_class,
*(loc:@cnn/bn_2_1/batch_normalization/beta*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
(cnn/bn_2_1/batch_normalization/beta/readIdentity#cnn/bn_2_1/batch_normalization/beta*
T0*6
_class,
*(loc:@cnn/bn_2_1/batch_normalization/beta*
_output_shapes	
:?
?
<cnn/bn_2_1/batch_normalization/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*=
_class3
1/loc:@cnn/bn_2_1/batch_normalization/moving_mean
?
*cnn/bn_2_1/batch_normalization/moving_mean
VariableV2*
shape:?*
shared_name *
dtype0*
	container *=
_class3
1/loc:@cnn/bn_2_1/batch_normalization/moving_mean*
_output_shapes	
:?
?
1cnn/bn_2_1/batch_normalization/moving_mean/AssignAssign*cnn/bn_2_1/batch_normalization/moving_mean<cnn/bn_2_1/batch_normalization/moving_mean/Initializer/zeros*
_output_shapes	
:?*=
_class3
1/loc:@cnn/bn_2_1/batch_normalization/moving_mean*
validate_shape(*
use_locking(*
T0
?
/cnn/bn_2_1/batch_normalization/moving_mean/readIdentity*cnn/bn_2_1/batch_normalization/moving_mean*=
_class3
1/loc:@cnn/bn_2_1/batch_normalization/moving_mean*
_output_shapes	
:?*
T0
?
?cnn/bn_2_1/batch_normalization/moving_variance/Initializer/onesConst*A
_class7
53loc:@cnn/bn_2_1/batch_normalization/moving_variance*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
.cnn/bn_2_1/batch_normalization/moving_variance
VariableV2*
shared_name *
shape:?*
dtype0*A
_class7
53loc:@cnn/bn_2_1/batch_normalization/moving_variance*
	container *
_output_shapes	
:?
?
5cnn/bn_2_1/batch_normalization/moving_variance/AssignAssign.cnn/bn_2_1/batch_normalization/moving_variance?cnn/bn_2_1/batch_normalization/moving_variance/Initializer/ones*
_output_shapes	
:?*
validate_shape(*
T0*A
_class7
53loc:@cnn/bn_2_1/batch_normalization/moving_variance*
use_locking(
?
3cnn/bn_2_1/batch_normalization/moving_variance/readIdentity.cnn/bn_2_1/batch_normalization/moving_variance*A
_class7
53loc:@cnn/bn_2_1/batch_normalization/moving_variance*
_output_shapes	
:?*
T0
?
-cnn/bn_2_1/batch_normalization/FusedBatchNormFusedBatchNormcnn/conv1d_2_1/conv2d/Conv2D)cnn/bn_2_1/batch_normalization/gamma/read(cnn/bn_2_1/batch_normalization/beta/read/cnn/bn_2_1/batch_normalization/moving_mean/read3cnn/bn_2_1/batch_normalization/moving_variance/read*
is_training( *
epsilon%o?:*
T0*L
_output_shapes:
8:?????????<?:?:?:?:?*
data_formatNHWC
i
$cnn/bn_2_1/batch_normalization/ConstConst*
valueB
 *?p}?*
_output_shapes
: *
dtype0
~
cnn/relu_2_1Relu-cnn/bn_2_1/batch_normalization/FusedBatchNorm*0
_output_shapes
:?????????<?*
T0
?
?cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?   ?   *
_output_shapes
:*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel
?
>cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*
_output_shapes
: 
?
@cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *6?=*
dtype0*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel
?
Icnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/shape*

seed *
T0*
seed2 */
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*
dtype0*(
_output_shapes
:??
?
=cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/mulMulIcnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal@cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/stddev*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*
T0*(
_output_shapes
:??
?
9cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normalAdd=cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/mul>cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal/mean*
T0*(
_output_shapes
:??*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel
?
cnn/conv1d_2_2/conv2d/kernel
VariableV2*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*
	container *
shape:??*(
_output_shapes
:??*
dtype0*
shared_name 
?
#cnn/conv1d_2_2/conv2d/kernel/AssignAssigncnn/conv1d_2_2/conv2d/kernel9cnn/conv1d_2_2/conv2d/kernel/Initializer/truncated_normal*
use_locking(*
validate_shape(*
T0*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*(
_output_shapes
:??
?
!cnn/conv1d_2_2/conv2d/kernel/readIdentitycnn/conv1d_2_2/conv2d/kernel*/
_class%
#!loc:@cnn/conv1d_2_2/conv2d/kernel*(
_output_shapes
:??*
T0
t
#cnn/conv1d_2_2/conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
cnn/conv1d_2_2/conv2d/Conv2DConv2Dcnn/relu_2_1!cnn/conv1d_2_2/conv2d/kernel/read*
	dilations
*
data_formatNHWC*0
_output_shapes
:?????????<?*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
5cnn/bn_2_2/batch_normalization/gamma/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0*7
_class-
+)loc:@cnn/bn_2_2/batch_normalization/gamma
?
$cnn/bn_2_2/batch_normalization/gamma
VariableV2*
shape:?*7
_class-
+)loc:@cnn/bn_2_2/batch_normalization/gamma*
shared_name *
dtype0*
	container *
_output_shapes	
:?
?
+cnn/bn_2_2/batch_normalization/gamma/AssignAssign$cnn/bn_2_2/batch_normalization/gamma5cnn/bn_2_2/batch_normalization/gamma/Initializer/ones*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@cnn/bn_2_2/batch_normalization/gamma*
_output_shapes	
:?
?
)cnn/bn_2_2/batch_normalization/gamma/readIdentity$cnn/bn_2_2/batch_normalization/gamma*
T0*7
_class-
+)loc:@cnn/bn_2_2/batch_normalization/gamma*
_output_shapes	
:?
?
5cnn/bn_2_2/batch_normalization/beta/Initializer/zerosConst*
dtype0*6
_class,
*(loc:@cnn/bn_2_2/batch_normalization/beta*
_output_shapes	
:?*
valueB?*    
?
#cnn/bn_2_2/batch_normalization/beta
VariableV2*
shape:?*6
_class,
*(loc:@cnn/bn_2_2/batch_normalization/beta*
	container *
shared_name *
dtype0*
_output_shapes	
:?
?
*cnn/bn_2_2/batch_normalization/beta/AssignAssign#cnn/bn_2_2/batch_normalization/beta5cnn/bn_2_2/batch_normalization/beta/Initializer/zeros*
T0*6
_class,
*(loc:@cnn/bn_2_2/batch_normalization/beta*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
(cnn/bn_2_2/batch_normalization/beta/readIdentity#cnn/bn_2_2/batch_normalization/beta*
_output_shapes	
:?*6
_class,
*(loc:@cnn/bn_2_2/batch_normalization/beta*
T0
?
<cnn/bn_2_2/batch_normalization/moving_mean/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*=
_class3
1/loc:@cnn/bn_2_2/batch_normalization/moving_mean*
dtype0
?
*cnn/bn_2_2/batch_normalization/moving_mean
VariableV2*
dtype0*
shared_name *
	container *=
_class3
1/loc:@cnn/bn_2_2/batch_normalization/moving_mean*
_output_shapes	
:?*
shape:?
?
1cnn/bn_2_2/batch_normalization/moving_mean/AssignAssign*cnn/bn_2_2/batch_normalization/moving_mean<cnn/bn_2_2/batch_normalization/moving_mean/Initializer/zeros*=
_class3
1/loc:@cnn/bn_2_2/batch_normalization/moving_mean*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
/cnn/bn_2_2/batch_normalization/moving_mean/readIdentity*cnn/bn_2_2/batch_normalization/moving_mean*=
_class3
1/loc:@cnn/bn_2_2/batch_normalization/moving_mean*
_output_shapes	
:?*
T0
?
?cnn/bn_2_2/batch_normalization/moving_variance/Initializer/onesConst*
dtype0*A
_class7
53loc:@cnn/bn_2_2/batch_normalization/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
.cnn/bn_2_2/batch_normalization/moving_variance
VariableV2*
	container *
_output_shapes	
:?*
shape:?*
shared_name *A
_class7
53loc:@cnn/bn_2_2/batch_normalization/moving_variance*
dtype0
?
5cnn/bn_2_2/batch_normalization/moving_variance/AssignAssign.cnn/bn_2_2/batch_normalization/moving_variance?cnn/bn_2_2/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*A
_class7
53loc:@cnn/bn_2_2/batch_normalization/moving_variance*
_output_shapes	
:?*
validate_shape(
?
3cnn/bn_2_2/batch_normalization/moving_variance/readIdentity.cnn/bn_2_2/batch_normalization/moving_variance*
T0*
_output_shapes	
:?*A
_class7
53loc:@cnn/bn_2_2/batch_normalization/moving_variance
?
-cnn/bn_2_2/batch_normalization/FusedBatchNormFusedBatchNormcnn/conv1d_2_2/conv2d/Conv2D)cnn/bn_2_2/batch_normalization/gamma/read(cnn/bn_2_2/batch_normalization/beta/read/cnn/bn_2_2/batch_normalization/moving_mean/read3cnn/bn_2_2/batch_normalization/moving_variance/read*
T0*
is_training( *
epsilon%o?:*L
_output_shapes:
8:?????????<?:?:?:?:?*
data_formatNHWC
i
$cnn/bn_2_2/batch_normalization/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *?p}?
~
cnn/relu_2_2Relu-cnn/bn_2_2/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:?????????<?
?
?cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*%
valueB"      ?      
?
>cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel
?
@cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *6?=*
_output_shapes
: *
dtype0*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel
?
Icnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/shape*
T0*
seed2 *

seed *
dtype0*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*(
_output_shapes
:??
?
=cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/mulMulIcnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/TruncatedNormal@cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/stddev*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*
T0*(
_output_shapes
:??
?
9cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normalAdd=cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/mul>cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal/mean*(
_output_shapes
:??*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*
T0
?
cnn/conv1d_2_3/conv2d/kernel
VariableV2*
shape:??*
	container */
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*(
_output_shapes
:??*
dtype0*
shared_name 
?
#cnn/conv1d_2_3/conv2d/kernel/AssignAssigncnn/conv1d_2_3/conv2d/kernel9cnn/conv1d_2_3/conv2d/kernel/Initializer/truncated_normal*
T0*
use_locking(*(
_output_shapes
:??*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel*
validate_shape(
?
!cnn/conv1d_2_3/conv2d/kernel/readIdentitycnn/conv1d_2_3/conv2d/kernel*
T0*(
_output_shapes
:??*/
_class%
#!loc:@cnn/conv1d_2_3/conv2d/kernel
t
#cnn/conv1d_2_3/conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
cnn/conv1d_2_3/conv2d/Conv2DConv2Dcnn/relu_2_2!cnn/conv1d_2_3/conv2d/kernel/read*
data_formatNHWC*
use_cudnn_on_gpu(*
	dilations
*
paddingSAME*
T0*0
_output_shapes
:?????????<?*
strides

?
5cnn/bn_2_3/batch_normalization/gamma/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*7
_class-
+)loc:@cnn/bn_2_3/batch_normalization/gamma*
dtype0
?
$cnn/bn_2_3/batch_normalization/gamma
VariableV2*
_output_shapes	
:?*
	container *
shape:?*
shared_name *
dtype0*7
_class-
+)loc:@cnn/bn_2_3/batch_normalization/gamma
?
+cnn/bn_2_3/batch_normalization/gamma/AssignAssign$cnn/bn_2_3/batch_normalization/gamma5cnn/bn_2_3/batch_normalization/gamma/Initializer/ones*
T0*7
_class-
+)loc:@cnn/bn_2_3/batch_normalization/gamma*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
)cnn/bn_2_3/batch_normalization/gamma/readIdentity$cnn/bn_2_3/batch_normalization/gamma*7
_class-
+)loc:@cnn/bn_2_3/batch_normalization/gamma*
_output_shapes	
:?*
T0
?
5cnn/bn_2_3/batch_normalization/beta/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *6
_class,
*(loc:@cnn/bn_2_3/batch_normalization/beta
?
#cnn/bn_2_3/batch_normalization/beta
VariableV2*
_output_shapes	
:?*
dtype0*
shared_name *
	container *6
_class,
*(loc:@cnn/bn_2_3/batch_normalization/beta*
shape:?
?
*cnn/bn_2_3/batch_normalization/beta/AssignAssign#cnn/bn_2_3/batch_normalization/beta5cnn/bn_2_3/batch_normalization/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*6
_class,
*(loc:@cnn/bn_2_3/batch_normalization/beta
?
(cnn/bn_2_3/batch_normalization/beta/readIdentity#cnn/bn_2_3/batch_normalization/beta*
T0*6
_class,
*(loc:@cnn/bn_2_3/batch_normalization/beta*
_output_shapes	
:?
?
<cnn/bn_2_3/batch_normalization/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*=
_class3
1/loc:@cnn/bn_2_3/batch_normalization/moving_mean*
_output_shapes	
:?
?
*cnn/bn_2_3/batch_normalization/moving_mean
VariableV2*
shape:?*
_output_shapes	
:?*
	container *
dtype0*=
_class3
1/loc:@cnn/bn_2_3/batch_normalization/moving_mean*
shared_name 
?
1cnn/bn_2_3/batch_normalization/moving_mean/AssignAssign*cnn/bn_2_3/batch_normalization/moving_mean<cnn/bn_2_3/batch_normalization/moving_mean/Initializer/zeros*=
_class3
1/loc:@cnn/bn_2_3/batch_normalization/moving_mean*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
/cnn/bn_2_3/batch_normalization/moving_mean/readIdentity*cnn/bn_2_3/batch_normalization/moving_mean*
T0*=
_class3
1/loc:@cnn/bn_2_3/batch_normalization/moving_mean*
_output_shapes	
:?
?
?cnn/bn_2_3/batch_normalization/moving_variance/Initializer/onesConst*A
_class7
53loc:@cnn/bn_2_3/batch_normalization/moving_variance*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
.cnn/bn_2_3/batch_normalization/moving_variance
VariableV2*
	container *
dtype0*
_output_shapes	
:?*A
_class7
53loc:@cnn/bn_2_3/batch_normalization/moving_variance*
shape:?*
shared_name 
?
5cnn/bn_2_3/batch_normalization/moving_variance/AssignAssign.cnn/bn_2_3/batch_normalization/moving_variance?cnn/bn_2_3/batch_normalization/moving_variance/Initializer/ones*
use_locking(*
_output_shapes	
:?*A
_class7
53loc:@cnn/bn_2_3/batch_normalization/moving_variance*
T0*
validate_shape(
?
3cnn/bn_2_3/batch_normalization/moving_variance/readIdentity.cnn/bn_2_3/batch_normalization/moving_variance*
_output_shapes	
:?*A
_class7
53loc:@cnn/bn_2_3/batch_normalization/moving_variance*
T0
?
-cnn/bn_2_3/batch_normalization/FusedBatchNormFusedBatchNormcnn/conv1d_2_3/conv2d/Conv2D)cnn/bn_2_3/batch_normalization/gamma/read(cnn/bn_2_3/batch_normalization/beta/read/cnn/bn_2_3/batch_normalization/moving_mean/read3cnn/bn_2_3/batch_normalization/moving_variance/read*
is_training( *
T0*
data_formatNHWC*L
_output_shapes:
8:?????????<?:?:?:?:?*
epsilon%o?:
i
$cnn/bn_2_3/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?
~
cnn/relu_2_3Relu-cnn/bn_2_3/batch_normalization/FusedBatchNorm*
T0*0
_output_shapes
:?????????<?
?
%cnn/maxpool1d_2/max_pooling2d/MaxPoolMaxPoolcnn/relu_2_3*
data_formatNHWC*0
_output_shapes
:??????????*
T0*
strides
*
paddingSAME*
ksize

x
cnn/flatten_2/ShapeShape%cnn/maxpool1d_2/max_pooling2d/MaxPool*
out_type0*
T0*
_output_shapes
:
k
!cnn/flatten_2/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
m
#cnn/flatten_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
m
#cnn/flatten_2/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
cnn/flatten_2/strided_sliceStridedSlicecnn/flatten_2/Shape!cnn/flatten_2/strided_slice/stack#cnn/flatten_2/strided_slice/stack_1#cnn/flatten_2/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
end_mask *
new_axis_mask *
_output_shapes
: *
Index0*
T0*
ellipsis_mask 
h
cnn/flatten_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
cnn/flatten_2/Reshape/shapePackcnn/flatten_2/strided_slicecnn/flatten_2/Reshape/shape/1*
T0*
N*
_output_shapes
:*

axis 
?
cnn/flatten_2/ReshapeReshape%cnn/maxpool1d_2/max_pooling2d/MaxPoolcnn/flatten_2/Reshape/shape*(
_output_shapes
:??????????*
T0*
Tshape0
s
drop_2/cond/SwitchSwitchplaceholders/is_trainingplaceholders/is_training*
_output_shapes
: : *
T0

W
drop_2/cond/switch_tIdentitydrop_2/cond/Switch:1*
T0
*
_output_shapes
: 
U
drop_2/cond/switch_fIdentitydrop_2/cond/Switch*
_output_shapes
: *
T0

Z
drop_2/cond/pred_idIdentityplaceholders/is_training*
_output_shapes
: *
T0

t
drop_2/cond/dropout/rateConst^drop_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *   ?
{
drop_2/cond/dropout/ShapeShape"drop_2/cond/dropout/Shape/Switch:1*
T0*
_output_shapes
:*
out_type0
?
 drop_2/cond/dropout/Shape/SwitchSwitchcnn/flatten_2/Reshapedrop_2/cond/pred_id*
T0*(
_class
loc:@cnn/flatten_2/Reshape*<
_output_shapes*
(:??????????:??????????
u
drop_2/cond/dropout/sub/xConst^drop_2/cond/switch_t*
valueB
 *  ??*
_output_shapes
: *
dtype0
t
drop_2/cond/dropout/subSubdrop_2/cond/dropout/sub/xdrop_2/cond/dropout/rate*
T0*
_output_shapes
: 
?
&drop_2/cond/dropout/random_uniform/minConst^drop_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
&drop_2/cond/dropout/random_uniform/maxConst^drop_2/cond/switch_t*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
0drop_2/cond/dropout/random_uniform/RandomUniformRandomUniformdrop_2/cond/dropout/Shape*(
_output_shapes
:??????????*
seed2 *
dtype0*

seed *
T0
?
&drop_2/cond/dropout/random_uniform/subSub&drop_2/cond/dropout/random_uniform/max&drop_2/cond/dropout/random_uniform/min*
T0*
_output_shapes
: 
?
&drop_2/cond/dropout/random_uniform/mulMul0drop_2/cond/dropout/random_uniform/RandomUniform&drop_2/cond/dropout/random_uniform/sub*(
_output_shapes
:??????????*
T0
?
"drop_2/cond/dropout/random_uniformAdd&drop_2/cond/dropout/random_uniform/mul&drop_2/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:??????????
?
drop_2/cond/dropout/addAdddrop_2/cond/dropout/sub"drop_2/cond/dropout/random_uniform*
T0*(
_output_shapes
:??????????
n
drop_2/cond/dropout/FloorFloordrop_2/cond/dropout/add*(
_output_shapes
:??????????*
T0
?
drop_2/cond/dropout/truedivRealDiv"drop_2/cond/dropout/Shape/Switch:1drop_2/cond/dropout/sub*
T0*(
_output_shapes
:??????????
?
drop_2/cond/dropout/mulMuldrop_2/cond/dropout/truedivdrop_2/cond/dropout/Floor*(
_output_shapes
:??????????*
T0
p
drop_2/cond/IdentityIdentitydrop_2/cond/Identity/Switch*(
_output_shapes
:??????????*
T0
?
drop_2/cond/Identity/SwitchSwitchcnn/flatten_2/Reshapedrop_2/cond/pred_id*<
_output_shapes*
(:??????????:??????????*
T0*(
_class
loc:@cnn/flatten_2/Reshape
?
drop_2/cond/MergeMergedrop_2/cond/Identitydrop_2/cond/dropout/mul**
_output_shapes
:??????????: *
T0*
N
q
rnn/reshape_seq_inputs/shapeConst*!
valueB"????      *
dtype0*
_output_shapes
:
?
rnn/reshape_seq_inputsReshapedrop_2/cond/Mergernn/reshape_seq_inputs/shape*,
_output_shapes
:??????????*
T0*
Tshape0
p
rnn/cond/SwitchSwitchplaceholders/is_trainingplaceholders/is_training*
T0
*
_output_shapes
: : 
Q
rnn/cond/switch_tIdentityrnn/cond/Switch:1*
_output_shapes
: *
T0

O
rnn/cond/switch_fIdentityrnn/cond/Switch*
T0
*
_output_shapes
: 
W
rnn/cond/pred_idIdentityplaceholders/is_training*
_output_shapes
: *
T0

g
rnn/cond/ConstConst^rnn/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *   ?
i
rnn/cond/Const_1Const^rnn/cond/switch_f*
dtype0*
valueB
 *  ??*
_output_shapes
: 
e
rnn/cond/MergeMergernn/cond/Const_1rnn/cond/Const*
_output_shapes
: : *
N*
T0
a
rnn/DropoutWrapperInit/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
c
rnn/DropoutWrapperInit/Const_1Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
valueB:?*
_output_shapes
:*
dtype0
?
Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Jrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/ConstKrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
_output_shapes
:*
T0*
N*

Tidx0
?
Ornn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
Irnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillJrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concatOrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0*
_output_shapes
:	?
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
_output_shapes
:*
valueB:*
dtype0
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
valueB:?*
_output_shapes
:*
dtype0
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
_output_shapes
:*
dtype0
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_5Const*
_output_shapes
:*
dtype0*
valueB:?
?
Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
Lrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_4Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_5Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
_output_shapes
:*
N
?
Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillLrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Qrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0*
_output_shapes
:	?
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_6Const*
dtype0*
valueB:*
_output_shapes
:
?
Krnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/Const_7Const*
dtype0*
valueB:?*
_output_shapes
:
N
rnn/rnn/RankConst*
dtype0*
value	B :*
_output_shapes
: 
U
rnn/rnn/range/startConst*
_output_shapes
: *
value	B :*
dtype0
U
rnn/rnn/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
v
rnn/rnn/rangeRangernn/rnn/range/startrnn/rnn/Rankrnn/rnn/range/delta*
_output_shapes
:*

Tidx0
h
rnn/rnn/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       
U
rnn/rnn/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
rnn/rnn/concatConcatV2rnn/rnn/concat/values_0rnn/rnn/rangernn/rnn/concat/axis*
T0*
N*

Tidx0*
_output_shapes
:
?
rnn/rnn/transpose	Transposernn/reshape_seq_inputsrnn/rnn/concat*
T0*
Tperm0*,
_output_shapes
:??????????
k
rnn/rnn/sequence_lengthIdentityplaceholders/seq_lengths*#
_output_shapes
:?????????*
T0
^
rnn/rnn/ShapeShapernn/rnn/transpose*
T0*
_output_shapes
:*
out_type0
e
rnn/rnn/strided_slice/stackConst*
valueB:*
_output_shapes
:*
dtype0
g
rnn/rnn/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
g
rnn/rnn/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
rnn/rnn/strided_sliceStridedSlicernn/rnn/Shapernn/rnn/strided_slice/stackrnn/rnn/strided_slice/stack_1rnn/rnn/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
end_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
_output_shapes
: 
f
rnn/rnn/Shape_1Shapernn/rnn/sequence_length*
T0*
out_type0*
_output_shapes
:
f
rnn/rnn/stackPackrnn/rnn/strided_slice*
T0*
_output_shapes
:*
N*

axis 
[
rnn/rnn/EqualEqualrnn/rnn/Shape_1rnn/rnn/stack*
T0*
_output_shapes
:
W
rnn/rnn/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
rnn/rnn/AllAllrnn/rnn/Equalrnn/rnn/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
?
rnn/rnn/Assert/ConstConst*
dtype0*
_output_shapes
: *H
value?B= B7Expected shape for Tensor rnn/rnn/sequence_length:0 is 
g
rnn/rnn/Assert/Const_1Const*!
valueB B but saw shape: *
_output_shapes
: *
dtype0
?
rnn/rnn/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *H
value?B= B7Expected shape for Tensor rnn/rnn/sequence_length:0 is 
m
rnn/rnn/Assert/Assert/data_2Const*
_output_shapes
: *!
valueB B but saw shape: *
dtype0
?
rnn/rnn/Assert/AssertAssertrnn/rnn/Allrnn/rnn/Assert/Assert/data_0rnn/rnn/stackrnn/rnn/Assert/Assert/data_2rnn/rnn/Shape_1*
	summarize*
T
2
~
rnn/rnn/CheckSeqLenIdentityrnn/rnn/sequence_length^rnn/rnn/Assert/Assert*#
_output_shapes
:?????????*
T0
`
rnn/rnn/Shape_2Shapernn/rnn/transpose*
_output_shapes
:*
T0*
out_type0
g
rnn/rnn/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
i
rnn/rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
i
rnn/rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
rnn/rnn/strided_slice_1StridedSlicernn/rnn/Shape_2rnn/rnn/strided_slice_1/stackrnn/rnn/strided_slice_1/stack_1rnn/rnn/strided_slice_1/stack_2*
_output_shapes
: *
Index0*
shrink_axis_mask*
T0*
new_axis_mask *
ellipsis_mask *

begin_mask *
end_mask 
`
rnn/rnn/Shape_3Shapernn/rnn/transpose*
_output_shapes
:*
T0*
out_type0
g
rnn/rnn/strided_slice_2/stackConst*
_output_shapes
:*
valueB:*
dtype0
i
rnn/rnn/strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
i
rnn/rnn/strided_slice_2/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
?
rnn/rnn/strided_slice_2StridedSlicernn/rnn/Shape_3rnn/rnn/strided_slice_2/stackrnn/rnn/strided_slice_2/stack_1rnn/rnn/strided_slice_2/stack_2*
ellipsis_mask *
T0*
Index0*
end_mask *
shrink_axis_mask*
new_axis_mask *
_output_shapes
: *

begin_mask 
X
rnn/rnn/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
rnn/rnn/ExpandDims
ExpandDimsrnn/rnn/strided_slice_2rnn/rnn/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
Z
rnn/rnn/Const_1Const*
_output_shapes
:*
valueB:?*
dtype0
W
rnn/rnn/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
rnn/rnn/concat_1ConcatV2rnn/rnn/ExpandDimsrnn/rnn/Const_1rnn/rnn/concat_1/axis*

Tidx0*
N*
T0*
_output_shapes
:
X
rnn/rnn/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
rnn/rnn/zerosFillrnn/rnn/concat_1rnn/rnn/zeros/Const*

index_type0*(
_output_shapes
:??????????*
T0
Y
rnn/rnn/Const_2Const*
valueB: *
_output_shapes
:*
dtype0
v
rnn/rnn/MinMinrnn/rnn/CheckSeqLenrnn/rnn/Const_2*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
Y
rnn/rnn/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
v
rnn/rnn/MaxMaxrnn/rnn/CheckSeqLenrnn/rnn/Const_3*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
N
rnn/rnn/timeConst*
dtype0*
_output_shapes
: *
value	B : 
?
rnn/rnn/TensorArrayTensorArrayV3rnn/rnn/strided_slice_1*
clear_after_read(*%
element_shape:??????????*
identical_element_shapes(*
dtype0*
dynamic_size( *3
tensor_array_namernn/rnn/dynamic_rnn/output_0*
_output_shapes

:: 
?
rnn/rnn/TensorArray_1TensorArrayV3rnn/rnn/strided_slice_1*
identical_element_shapes(*
clear_after_read(*2
tensor_array_namernn/rnn/dynamic_rnn/input_0*
_output_shapes

:: *%
element_shape:??????????*
dtype0*
dynamic_size( 
q
 rnn/rnn/TensorArrayUnstack/ShapeShapernn/rnn/transpose*
_output_shapes
:*
T0*
out_type0
x
.rnn/rnn/TensorArrayUnstack/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
z
0rnn/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
z
0rnn/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
(rnn/rnn/TensorArrayUnstack/strided_sliceStridedSlice rnn/rnn/TensorArrayUnstack/Shape.rnn/rnn/TensorArrayUnstack/strided_slice/stack0rnn/rnn/TensorArrayUnstack/strided_slice/stack_10rnn/rnn/TensorArrayUnstack/strided_slice/stack_2*
end_mask *
shrink_axis_mask*
Index0*
new_axis_mask *
ellipsis_mask *

begin_mask *
T0*
_output_shapes
: 
h
&rnn/rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
h
&rnn/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
?
 rnn/rnn/TensorArrayUnstack/rangeRange&rnn/rnn/TensorArrayUnstack/range/start(rnn/rnn/TensorArrayUnstack/strided_slice&rnn/rnn/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????*

Tidx0
?
Brnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/rnn/TensorArray_1 rnn/rnn/TensorArrayUnstack/rangernn/rnn/transposernn/rnn/TensorArray_1:1*
T0*$
_class
loc:@rnn/rnn/transpose*
_output_shapes
: 
S
rnn/rnn/Maximum/xConst*
_output_shapes
: *
value	B :*
dtype0
[
rnn/rnn/MaximumMaximumrnn/rnn/Maximum/xrnn/rnn/Max*
T0*
_output_shapes
: 
e
rnn/rnn/MinimumMinimumrnn/rnn/strided_slice_1rnn/rnn/Maximum*
T0*
_output_shapes
: 
a
rnn/rnn/while/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
?
rnn/rnn/while/EnterEnterrnn/rnn/while/iteration_counter*+

frame_namernn/rnn/while/while_context*
is_constant( *
T0*
_output_shapes
: *
parallel_iterations 
?
rnn/rnn/while/Enter_1Enterrnn/rnn/time*
is_constant( *
T0*
parallel_iterations *
_output_shapes
: *+

frame_namernn/rnn/while/while_context
?
rnn/rnn/while/Enter_2Enterrnn/rnn/TensorArray:1*
is_constant( *
T0*+

frame_namernn/rnn/while/while_context*
parallel_iterations *
_output_shapes
: 
?
rnn/rnn/while/Enter_3EnterIrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros*+

frame_namernn/rnn/while/while_context*
_output_shapes
:	?*
is_constant( *
T0*
parallel_iterations 
?
rnn/rnn/while/Enter_4EnterKrnn/MultiRNNCellZeroState/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*+

frame_namernn/rnn/while/while_context*
T0*
parallel_iterations *
_output_shapes
:	?*
is_constant( 
z
rnn/rnn/while/MergeMergernn/rnn/while/Enterrnn/rnn/while/NextIteration*
N*
_output_shapes
: : *
T0
?
rnn/rnn/while/Merge_1Mergernn/rnn/while/Enter_1rnn/rnn/while/NextIteration_1*
_output_shapes
: : *
T0*
N
?
rnn/rnn/while/Merge_2Mergernn/rnn/while/Enter_2rnn/rnn/while/NextIteration_2*
N*
T0*
_output_shapes
: : 
?
rnn/rnn/while/Merge_3Mergernn/rnn/while/Enter_3rnn/rnn/while/NextIteration_3*
N*
T0*!
_output_shapes
:	?: 
?
rnn/rnn/while/Merge_4Mergernn/rnn/while/Enter_4rnn/rnn/while/NextIteration_4*!
_output_shapes
:	?: *
T0*
N
j
rnn/rnn/while/LessLessrnn/rnn/while/Mergernn/rnn/while/Less/Enter*
T0*
_output_shapes
: 
?
rnn/rnn/while/Less/EnterEnterrnn/rnn/strided_slice_1*
is_constant(*+

frame_namernn/rnn/while/while_context*
parallel_iterations *
T0*
_output_shapes
: 
p
rnn/rnn/while/Less_1Lessrnn/rnn/while/Merge_1rnn/rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
?
rnn/rnn/while/Less_1/EnterEnterrnn/rnn/Minimum*
_output_shapes
: *
is_constant(*+

frame_namernn/rnn/while/while_context*
T0*
parallel_iterations 
h
rnn/rnn/while/LogicalAnd
LogicalAndrnn/rnn/while/Lessrnn/rnn/while/Less_1*
_output_shapes
: 
T
rnn/rnn/while/LoopCondLoopCondrnn/rnn/while/LogicalAnd*
_output_shapes
: 
?
rnn/rnn/while/SwitchSwitchrnn/rnn/while/Mergernn/rnn/while/LoopCond*
T0*&
_class
loc:@rnn/rnn/while/Merge*
_output_shapes
: : 
?
rnn/rnn/while/Switch_1Switchrnn/rnn/while/Merge_1rnn/rnn/while/LoopCond*
_output_shapes
: : *
T0*(
_class
loc:@rnn/rnn/while/Merge_1
?
rnn/rnn/while/Switch_2Switchrnn/rnn/while/Merge_2rnn/rnn/while/LoopCond*
T0*
_output_shapes
: : *(
_class
loc:@rnn/rnn/while/Merge_2
?
rnn/rnn/while/Switch_3Switchrnn/rnn/while/Merge_3rnn/rnn/while/LoopCond*
T0**
_output_shapes
:	?:	?*(
_class
loc:@rnn/rnn/while/Merge_3
?
rnn/rnn/while/Switch_4Switchrnn/rnn/while/Merge_4rnn/rnn/while/LoopCond**
_output_shapes
:	?:	?*(
_class
loc:@rnn/rnn/while/Merge_4*
T0
[
rnn/rnn/while/IdentityIdentityrnn/rnn/while/Switch:1*
T0*
_output_shapes
: 
_
rnn/rnn/while/Identity_1Identityrnn/rnn/while/Switch_1:1*
T0*
_output_shapes
: 
_
rnn/rnn/while/Identity_2Identityrnn/rnn/while/Switch_2:1*
_output_shapes
: *
T0
h
rnn/rnn/while/Identity_3Identityrnn/rnn/while/Switch_3:1*
_output_shapes
:	?*
T0
h
rnn/rnn/while/Identity_4Identityrnn/rnn/while/Switch_4:1*
T0*
_output_shapes
:	?
n
rnn/rnn/while/add/yConst^rnn/rnn/while/Identity*
_output_shapes
: *
dtype0*
value	B :
f
rnn/rnn/while/addAddrnn/rnn/while/Identityrnn/rnn/while/add/y*
T0*
_output_shapes
: 
?
rnn/rnn/while/TensorArrayReadV3TensorArrayReadV3%rnn/rnn/while/TensorArrayReadV3/Enterrnn/rnn/while/Identity_1'rnn/rnn/while/TensorArrayReadV3/Enter_1*(
_output_shapes
:??????????*
dtype0
?
%rnn/rnn/while/TensorArrayReadV3/EnterEnterrnn/rnn/TensorArray_1*
T0*
is_constant(*+

frame_namernn/rnn/while/while_context*
_output_shapes
:*
parallel_iterations 
?
'rnn/rnn/while/TensorArrayReadV3/Enter_1EnterBrnn/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
is_constant(*
_output_shapes
: *+

frame_namernn/rnn/while/while_context*
T0*
parallel_iterations 
?
rnn/rnn/while/GreaterEqualGreaterEqualrnn/rnn/while/Identity_1 rnn/rnn/while/GreaterEqual/Enter*
T0*#
_output_shapes
:?????????
?
 rnn/rnn/while/GreaterEqual/EnterEnterrnn/rnn/CheckSeqLen*#
_output_shapes
:?????????*+

frame_namernn/rnn/while/while_context*
T0*
parallel_iterations *
is_constant(
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
?
Mrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?7?*
_output_shapes
: *A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
?
Mrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
valueB
 *?7=
?
Wrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
T0* 
_output_shapes
:
? ?*
seed2 *A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0*

seed 
?
Mrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubMrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxMrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0
?
Mrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMulWrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformMrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
? ?*
T0
?
Irnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddMrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
? ?
?
.rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelVarHandleOp*A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
: *
dtype0*
	container *
shape:
? ?*?
shared_name0.rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
_output_shapes
: 
?
5rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssignVariableOp.rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernelIrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
?
Brnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp.rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
? ?*A
_class7
53loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
dtype0
?
<rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Read/IdentityIdentityBrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Read/ReadVariableOp* 
_output_shapes
:
? ?*
T0
?
Nrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:?*?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
?
Drnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0
?
>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zerosFillNrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros/shape_as_tensorDrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros/Const*
T0*?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:?*

index_type0
?
,rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/biasVarHandleOp*?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
	container *
shape:?*
_output_shapes
: *
dtype0*=
shared_name.,rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
?
Mrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp,rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes
: 
?
3rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssignVariableOp,rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros*?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0
?
@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Read/ReadVariableOpReadVariableOp,rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
dtype0*
_output_shapes	
:?*?
_class5
31loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias
?
:rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Read/IdentityIdentity@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:?
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/shapeConst*
dtype0*
valueB:?*
_output_shapes
:*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *׳ݽ*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/maxConst*
valueB
 *׳?=*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
dtype0*
_output_shapes
: 
?
Yrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/RandomUniformRandomUniformQrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/shape*
dtype0*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
seed2 *
T0*

seed *
_output_shapes	
:?
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/subSubOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/maxOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/min*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
T0*
_output_shapes
: 
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/mulMulYrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/RandomUniformOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/sub*
_output_shapes	
:?*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
T0
?
Krnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniformAddOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/mulOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform/min*
T0*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
_output_shapes	
:?
?
0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diagVarHandleOp*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
dtype0*
	container *
shape:?*
_output_shapes
: *A
shared_name20rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/IsInitialized/VarIsInitializedOpVarIsInitializedOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
_output_shapes
: 
?
7rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/AssignAssignVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diagKrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Initializer/random_uniform*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
dtype0
?
Drnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Read/ReadVariableOpReadVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag*
dtype0*
_output_shapes	
:?
?
>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Read/IdentityIdentityDrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Read/ReadVariableOp*
_output_shapes	
:?*
T0
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
valueB:?
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/minConst*
valueB
 *׳ݽ*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
dtype0*
_output_shapes
: 
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *׳?=*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag
?
Yrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/RandomUniformRandomUniformQrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/shape*
_output_shapes	
:?*
seed2 *
dtype0*

seed *C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
T0
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/subSubOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/maxOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/min*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
_output_shapes
: *
T0
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/mulMulYrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/RandomUniformOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/sub*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
T0*
_output_shapes	
:?
?
Krnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniformAddOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/mulOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform/min*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
T0*
_output_shapes	
:?
?
0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diagVarHandleOp*A
shared_name20rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
shape:?*
	container *C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
_output_shapes
: *
dtype0
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/IsInitialized/VarIsInitializedOpVarIsInitializedOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
_output_shapes
: 
?
7rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/AssignAssignVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diagKrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Initializer/random_uniform*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
dtype0
?
Drnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Read/ReadVariableOpReadVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag*
_output_shapes	
:?*
dtype0
?
>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Read/IdentityIdentityDrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Read/ReadVariableOp*
T0*
_output_shapes	
:?
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/shapeConst*
valueB:?*
_output_shapes
:*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
dtype0
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *׳ݽ*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/maxConst*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
dtype0*
valueB
 *׳?=*
_output_shapes
: 
?
Yrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/RandomUniformRandomUniformQrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/shape*
seed2 *
T0*
dtype0*
_output_shapes	
:?*

seed *C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/subSubOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/maxOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/min*
_output_shapes
: *
T0*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag
?
Ornn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/mulMulYrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/RandomUniformOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/sub*
_output_shapes	
:?*
T0*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag
?
Krnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniformAddOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/mulOrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform/min*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
T0*
_output_shapes	
:?
?
0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diagVarHandleOp*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
shape:?*A
shared_name20rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
dtype0*
	container *
_output_shapes
: 
?
Qrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/IsInitialized/VarIsInitializedOpVarIsInitializedOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
_output_shapes
: 
?
7rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/AssignAssignVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diagKrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Initializer/random_uniform*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
dtype0
?
Drnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Read/ReadVariableOpReadVariableOp0rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
_output_shapes	
:?*C
_class9
75loc:@rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag*
dtype0
?
>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Read/IdentityIdentityDrnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Read/ReadVariableOp*
_output_shapes	
:?*
T0
?
=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axisConst^rnn/rnn/while/Identity*
dtype0*
_output_shapes
: *
value	B :
?
8rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatConcatV2rnn/rnn/while/TensorArrayReadV3rnn/rnn/while/Identity_4=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:	? 
?
8rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulMatMul8rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat>rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter*
transpose_b( *
_output_shapes
:	?*
T0*
transpose_a( 
?
>rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/EnterEnter<rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Read/Identity*
T0*+

frame_namernn/rnn/while/while_context*
is_constant(*
parallel_iterations * 
_output_shapes
:
? ?
?
9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAdd8rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul?rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	?
?
?rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter:rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Read/Identity*
_output_shapes	
:?*
T0*+

frame_namernn/rnn/while/while_context*
parallel_iterations *
is_constant(
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/ConstConst^rnn/rnn/while/Identity*
_output_shapes
: *
value	B :*
dtype0
?
Arnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^rnn/rnn/while/Identity*
dtype0*
value	B :*
_output_shapes
: 
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/splitSplitArnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*@
_output_shapes.
,:	?:	?:	?:	?*
	num_split*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/yConst^rnn/rnn/while/Identity*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/addAdd9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:27rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0*
_output_shapes
:	?
?
5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mulMul;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul/Enterrnn/rnn/while/Identity_3*
T0*
_output_shapes
:	?
?
;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul/EnterEnter>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_f_diag/Read/Identity*
parallel_iterations *+

frame_namernn/rnn/while/while_context*
is_constant(*
T0*
_output_shapes	
:?
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1Add5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul*
T0*
_output_shapes
:	?
?
9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoidrnn/rnn/while/Identity_3*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2/Enterrnn/rnn/while/Identity_3*
T0*
_output_shapes
:	?
?
=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2/EnterEnter>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_i_diag/Read/Identity*
T0*
parallel_iterations *+

frame_namernn/rnn/while/while_context*
is_constant(*
_output_shapes	
:?
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2Add7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
_output_shapes
:	?*
T0
?
;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2*
T0*
_output_shapes
:	?
?
6rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/TanhTanh9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_3Mul;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_16rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3Add7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_17rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_3*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4Mul=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4/Enter7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*
_output_shapes
:	?*
T0
?
=rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4/EnterEnter>rnn/rnn/multi_rnn_cell/cell_0/lstm_cell/w_o_diag/Read/Identity*
parallel_iterations *+

frame_namernn/rnn/while/while_context*
_output_shapes	
:?*
T0*
is_constant(
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_4Add9rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:37rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4*
_output_shapes
:	?*
T0
?
;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_4*
_output_shapes
:	?*
T0
?
8rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*
T0*
_output_shapes
:	?
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5Mul;rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_28rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*
_output_shapes
:	?
?
-rnn/rnn/while/rnn/multi_rnn_cell/cell_0/sub/xConst^rnn/rnn/while/Identity*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
+rnn/rnn/while/rnn/multi_rnn_cell/cell_0/subSub-rnn/rnn/while/rnn/multi_rnn_cell/cell_0/sub/x1rnn/rnn/while/rnn/multi_rnn_cell/cell_0/sub/Enter*
T0*
_output_shapes
: 
?
1rnn/rnn/while/rnn/multi_rnn_cell/cell_0/sub/EnterEnterrnn/cond/Merge*+

frame_namernn/rnn/while/while_context*
_output_shapes
: *
T0*
is_constant(*
parallel_iterations 
?
5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/ShapeConst^rnn/rnn/while/Identity*
_output_shapes
:*
valueB"      *
dtype0
?
5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/sub/xConst^rnn/rnn/while/Identity*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/subSub5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/sub/x+rnn/rnn/while/rnn/multi_rnn_cell/cell_0/sub*
T0*
_output_shapes
: 
?
Brnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/minConst^rnn/rnn/while/Identity*
_output_shapes
: *
dtype0*
valueB
 *    
?
Brnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxConst^rnn/rnn/while/Identity*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
Lrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformRandomUniform5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Shape*
dtype0*
T0*
_output_shapes
:	?*
seed2 *

seed 
?
Brnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/subSubBrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/maxBrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
: *
T0
?
Brnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulMulLrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/RandomUniformBrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/sub*
T0*
_output_shapes
:	?
?
>rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniformAddBrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/mulBrnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform/min*
_output_shapes
:	?*
T0
?
3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/addAdd3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/sub>rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/random_uniform*
T0*
_output_shapes
:	?
?
5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/FloorFloor3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/add*
_output_shapes
:	?*
T0
?
7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/truedivRealDiv7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_53rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/sub*
_output_shapes
:	?*
T0
?
3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mulMul7rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/truediv5rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/Floor*
_output_shapes
:	?*
T0
?
rnn/rnn/while/SelectSelectrnn/rnn/while/GreaterEqualrnn/rnn/while/Select/Enter3rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
_output_shapes
:	?*
T0*F
_class<
:8loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul
?
rnn/rnn/while/Select/EnterEnterrnn/rnn/zeros*+

frame_namernn/rnn/while/while_context*F
_class<
:8loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations *
is_constant(*
T0*(
_output_shapes
:??????????
?
rnn/rnn/while/Select_1Selectrnn/rnn/while/GreaterEqualrnn/rnn/while/Identity_37rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*J
_class@
><loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*
_output_shapes
:	?*
T0
?
rnn/rnn/while/Select_2Selectrnn/rnn/while/GreaterEqualrnn/rnn/while/Identity_47rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5*
T0*J
_class@
><loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5*
_output_shapes
:	?
?
1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV37rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/rnn/while/Identity_1rnn/rnn/while/Selectrnn/rnn/while/Identity_2*
_output_shapes
: *
T0*F
_class<
:8loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul
?
7rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/rnn/TensorArray*F
_class<
:8loc:@rnn/rnn/while/rnn/multi_rnn_cell/cell_0/dropout/mul*
parallel_iterations *+

frame_namernn/rnn/while/while_context*
T0*
is_constant(*
_output_shapes
:
p
rnn/rnn/while/add_1/yConst^rnn/rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
l
rnn/rnn/while/add_1Addrnn/rnn/while/Identity_1rnn/rnn/while/add_1/y*
_output_shapes
: *
T0
`
rnn/rnn/while/NextIterationNextIterationrnn/rnn/while/add*
_output_shapes
: *
T0
d
rnn/rnn/while/NextIteration_1NextIterationrnn/rnn/while/add_1*
T0*
_output_shapes
: 
?
rnn/rnn/while/NextIteration_2NextIteration1rnn/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
p
rnn/rnn/while/NextIteration_3NextIterationrnn/rnn/while/Select_1*
T0*
_output_shapes
:	?
p
rnn/rnn/while/NextIteration_4NextIterationrnn/rnn/while/Select_2*
_output_shapes
:	?*
T0
Q
rnn/rnn/while/ExitExitrnn/rnn/while/Switch*
_output_shapes
: *
T0
U
rnn/rnn/while/Exit_1Exitrnn/rnn/while/Switch_1*
T0*
_output_shapes
: 
U
rnn/rnn/while/Exit_2Exitrnn/rnn/while/Switch_2*
T0*
_output_shapes
: 
^
rnn/rnn/while/Exit_3Exitrnn/rnn/while/Switch_3*
_output_shapes
:	?*
T0
^
rnn/rnn/while/Exit_4Exitrnn/rnn/while/Switch_4*
_output_shapes
:	?*
T0
?
*rnn/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/rnn/TensorArrayrnn/rnn/while/Exit_2*&
_class
loc:@rnn/rnn/TensorArray*
_output_shapes
: 
?
$rnn/rnn/TensorArrayStack/range/startConst*
dtype0*
value	B : *&
_class
loc:@rnn/rnn/TensorArray*
_output_shapes
: 
?
$rnn/rnn/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *&
_class
loc:@rnn/rnn/TensorArray
?
rnn/rnn/TensorArrayStack/rangeRange$rnn/rnn/TensorArrayStack/range/start*rnn/rnn/TensorArrayStack/TensorArraySizeV3$rnn/rnn/TensorArrayStack/range/delta*#
_output_shapes
:?????????*&
_class
loc:@rnn/rnn/TensorArray*

Tidx0
?
,rnn/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/rnn/TensorArrayrnn/rnn/TensorArrayStack/rangernn/rnn/while/Exit_2*&
_class
loc:@rnn/rnn/TensorArray*#
_output_shapes
:?*
element_shape:	?*
dtype0
Z
rnn/rnn/Const_4Const*
_output_shapes
:*
valueB:?*
dtype0
P
rnn/rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
W
rnn/rnn/range_1/startConst*
_output_shapes
: *
value	B :*
dtype0
W
rnn/rnn/range_1/deltaConst*
value	B :*
_output_shapes
: *
dtype0
~
rnn/rnn/range_1Rangernn/rnn/range_1/startrnn/rnn/Rank_1rnn/rnn/range_1/delta*
_output_shapes
:*

Tidx0
j
rnn/rnn/concat_2/values_0Const*
valueB"       *
_output_shapes
:*
dtype0
W
rnn/rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
rnn/rnn/concat_2ConcatV2rnn/rnn/concat_2/values_0rnn/rnn/range_1rnn/rnn/concat_2/axis*

Tidx0*
T0*
N*
_output_shapes
:
?
rnn/rnn/transpose_1	Transpose,rnn/rnn/TensorArrayStack/TensorArrayGatherV3rnn/rnn/concat_2*#
_output_shapes
:?*
T0*
Tperm0
o
rnn/reshape_nonseq_input/shapeConst*
valueB"????   *
_output_shapes
:*
dtype0
?
rnn/reshape_nonseq_inputReshapernn/rnn/transpose_1rnn/reshape_nonseq_input/shape* 
_output_shapes
:
??*
Tshape0*
T0
?
>softmax_linear/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *.
_class$
" loc:@softmax_linear/dense/kernel*
dtype0*
_output_shapes
:
?
=softmax_linear/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: *.
_class$
" loc:@softmax_linear/dense/kernel
?
?softmax_linear/dense/kernel/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@softmax_linear/dense/kernel*
valueB
 *6??=*
_output_shapes
: *
dtype0
?
Hsoftmax_linear/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>softmax_linear/dense/kernel/Initializer/truncated_normal/shape*

seed *.
_class$
" loc:@softmax_linear/dense/kernel*
seed2 *
_output_shapes
:	?*
T0*
dtype0
?
<softmax_linear/dense/kernel/Initializer/truncated_normal/mulMulHsoftmax_linear/dense/kernel/Initializer/truncated_normal/TruncatedNormal?softmax_linear/dense/kernel/Initializer/truncated_normal/stddev*.
_class$
" loc:@softmax_linear/dense/kernel*
T0*
_output_shapes
:	?
?
8softmax_linear/dense/kernel/Initializer/truncated_normalAdd<softmax_linear/dense/kernel/Initializer/truncated_normal/mul=softmax_linear/dense/kernel/Initializer/truncated_normal/mean*
_output_shapes
:	?*
T0*.
_class$
" loc:@softmax_linear/dense/kernel
?
softmax_linear/dense/kernel
VariableV2*
	container *
shared_name *
dtype0*
shape:	?*.
_class$
" loc:@softmax_linear/dense/kernel*
_output_shapes
:	?
?
"softmax_linear/dense/kernel/AssignAssignsoftmax_linear/dense/kernel8softmax_linear/dense/kernel/Initializer/truncated_normal*
_output_shapes
:	?*.
_class$
" loc:@softmax_linear/dense/kernel*
T0*
use_locking(*
validate_shape(
?
 softmax_linear/dense/kernel/readIdentitysoftmax_linear/dense/kernel*
T0*.
_class$
" loc:@softmax_linear/dense/kernel*
_output_shapes
:	?
?
+softmax_linear/dense/bias/Initializer/ConstConst*
valueB*    *,
_class"
 loc:@softmax_linear/dense/bias*
dtype0*
_output_shapes
:
?
softmax_linear/dense/bias
VariableV2*
_output_shapes
:*
shared_name *,
_class"
 loc:@softmax_linear/dense/bias*
	container *
shape:*
dtype0
?
 softmax_linear/dense/bias/AssignAssignsoftmax_linear/dense/bias+softmax_linear/dense/bias/Initializer/Const*
validate_shape(*
T0*,
_class"
 loc:@softmax_linear/dense/bias*
use_locking(*
_output_shapes
:
?
softmax_linear/dense/bias/readIdentitysoftmax_linear/dense/bias*
_output_shapes
:*,
_class"
 loc:@softmax_linear/dense/bias*
T0
?
softmax_linear/dense/MatMulMatMulrnn/reshape_nonseq_input softmax_linear/dense/kernel/read*
_output_shapes
:	?*
transpose_a( *
transpose_b( *
T0
?
softmax_linear/dense/BiasAddBiasAddsoftmax_linear/dense/MatMulsoftmax_linear/dense/bias/read*
T0*
_output_shapes
:	?*
data_formatNHWC
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
?
ArgMaxArgMaxsoftmax_linear/dense/BiasAddArgMax/dimension*

Tidx0*
T0*
output_type0	*
_output_shapes	
:?
k
loss_ce_per_sample/ShapeShapeplaceholders/labels*
out_type0*
_output_shapes
:*
T0
?
%loss_ce_per_sample/loss_ce_per_sample#SparseSoftmaxCrossEntropyWithLogitssoftmax_linear/dense/BiasAddplaceholders/labels*&
_output_shapes
:?:	?*
Tlabels0*
T0

loss_ce_mean/MulMulplaceholders/loss_weights%loss_ce_per_sample/loss_ce_per_sample*
T0*
_output_shapes	
:?
b
loss_ce_mean/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
c
loss_ce_mean/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
loss_ce_mean/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
?
loss_ce_mean/one_hotOneHotplaceholders/labelsloss_ce_mean/one_hot/depthloss_ce_mean/one_hot/on_valueloss_ce_mean/one_hot/off_value*'
_output_shapes
:?????????*
axis?????????*
T0*
TI0
q
loss_ce_mean/Mul_1/yConst*)
value B"  ??  ??  ??  ??  ??*
dtype0*
_output_shapes
:
w
loss_ce_mean/Mul_1Mulloss_ce_mean/one_hotloss_ce_mean/Mul_1/y*'
_output_shapes
:?????????*
T0
d
"loss_ce_mean/Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
?
loss_ce_mean/SumSumloss_ce_mean/Mul_1"loss_ce_mean/Sum/reduction_indices*

Tidx0*#
_output_shapes
:?????????*
T0*
	keep_dims( 
c
loss_ce_mean/Mul_2Mulloss_ce_mean/Mulloss_ce_mean/Sum*
_output_shapes	
:?*
T0
\
loss_ce_mean/ConstConst*
valueB: *
_output_shapes
:*
dtype0

loss_ce_mean/Sum_1Sumloss_ce_mean/Mul_2loss_ce_mean/Const*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
^
loss_ce_mean/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
?
loss_ce_mean/Sum_2Sumplaceholders/loss_weightsloss_ce_mean/Const_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
h
loss_ce_mean/truedivRealDivloss_ce_mean/Sum_1loss_ce_mean/Sum_2*
T0*
_output_shapes
: 
R
L2LossL2Losscnn/conv1d_1/conv2d/kernel/read*
T0*
_output_shapes
: 
V
L2Loss_1L2Loss!cnn/conv1d_2_1/conv2d/kernel/read*
_output_shapes
: *
T0
V
L2Loss_2L2Loss!cnn/conv1d_2_2/conv2d/kernel/read*
_output_shapes
: *
T0
V
L2Loss_3L2Loss!cnn/conv1d_2_3/conv2d/kernel/read*
T0*
_output_shapes
: 
_
l2_lossAddNL2LossL2Loss_1L2Loss_2L2Loss_3*
_output_shapes
: *
N*
T0
J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:
;
MulMull2_lossMul/y*
T0*
_output_shapes
: 
F
addAddloss_ce_mean/truedivMul*
T0*
_output_shapes
: 
?
+stream_metrics/mean/total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@stream_metrics/mean/total
?
stream_metrics/mean/total
VariableV2*
	container *
dtype0*
shape: *
shared_name *,
_class"
 loc:@stream_metrics/mean/total*
_output_shapes
: 
?
 stream_metrics/mean/total/AssignAssignstream_metrics/mean/total+stream_metrics/mean/total/Initializer/zeros*,
_class"
 loc:@stream_metrics/mean/total*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
?
stream_metrics/mean/total/readIdentitystream_metrics/mean/total*
_output_shapes
: *
T0*,
_class"
 loc:@stream_metrics/mean/total
?
+stream_metrics/mean/count/Initializer/zerosConst*,
_class"
 loc:@stream_metrics/mean/count*
dtype0*
valueB
 *    *
_output_shapes
: 
?
stream_metrics/mean/count
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *
shared_name *,
_class"
 loc:@stream_metrics/mean/count
?
 stream_metrics/mean/count/AssignAssignstream_metrics/mean/count+stream_metrics/mean/count/Initializer/zeros*
use_locking(*,
_class"
 loc:@stream_metrics/mean/count*
_output_shapes
: *
T0*
validate_shape(
?
stream_metrics/mean/count/readIdentitystream_metrics/mean/count*
_output_shapes
: *,
_class"
 loc:@stream_metrics/mean/count*
T0
Z
stream_metrics/mean/SizeConst*
_output_shapes
: *
value	B :*
dtype0
}
stream_metrics/mean/ToFloatCaststream_metrics/mean/Size*

DstT0*

SrcT0*
_output_shapes
: *
Truncate( 
\
stream_metrics/mean/ConstConst*
valueB *
_output_shapes
: *
dtype0
|
stream_metrics/mean/SumSumaddstream_metrics/mean/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
?
stream_metrics/mean/AssignAdd	AssignAddstream_metrics/mean/totalstream_metrics/mean/Sum*
_output_shapes
: *
use_locking( *
T0*,
_class"
 loc:@stream_metrics/mean/total
?
stream_metrics/mean/AssignAdd_1	AssignAddstream_metrics/mean/countstream_metrics/mean/ToFloat^add*
_output_shapes
: *
T0*,
_class"
 loc:@stream_metrics/mean/count*
use_locking( 
b
stream_metrics/mean/Maximum/yConst*
_output_shapes
: *
valueB
 *    *
dtype0
?
stream_metrics/mean/MaximumMaximumstream_metrics/mean/count/readstream_metrics/mean/Maximum/y*
_output_shapes
: *
T0
?
stream_metrics/mean/valueDivNoNanstream_metrics/mean/total/readstream_metrics/mean/Maximum*
T0*
_output_shapes
: 
d
stream_metrics/mean/Maximum_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
stream_metrics/mean/Maximum_1Maximumstream_metrics/mean/AssignAdd_1stream_metrics/mean/Maximum_1/y*
T0*
_output_shapes
: 
?
stream_metrics/mean/update_opDivNoNanstream_metrics/mean/AssignAddstream_metrics/mean/Maximum_1*
T0*
_output_shapes
: 
h
stream_metrics/CastCastArgMax*
Truncate( *
_output_shapes	
:?*

SrcT0	*

DstT0
m
stream_metrics/EqualEqualstream_metrics/Castplaceholders/labels*
T0*
_output_shapes	
:?
y
stream_metrics/ToFloatCaststream_metrics/Equal*

SrcT0
*
Truncate( *

DstT0*
_output_shapes	
:?
?
/stream_metrics/accuracy/total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB
 *    *0
_class&
$"loc:@stream_metrics/accuracy/total
?
stream_metrics/accuracy/total
VariableV2*
	container *
_output_shapes
: *
dtype0*0
_class&
$"loc:@stream_metrics/accuracy/total*
shape: *
shared_name 
?
$stream_metrics/accuracy/total/AssignAssignstream_metrics/accuracy/total/stream_metrics/accuracy/total/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
: *0
_class&
$"loc:@stream_metrics/accuracy/total
?
"stream_metrics/accuracy/total/readIdentitystream_metrics/accuracy/total*0
_class&
$"loc:@stream_metrics/accuracy/total*
T0*
_output_shapes
: 
?
/stream_metrics/accuracy/count/Initializer/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    *0
_class&
$"loc:@stream_metrics/accuracy/count
?
stream_metrics/accuracy/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *0
_class&
$"loc:@stream_metrics/accuracy/count*
shared_name *
	container 
?
$stream_metrics/accuracy/count/AssignAssignstream_metrics/accuracy/count/stream_metrics/accuracy/count/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
: *0
_class&
$"loc:@stream_metrics/accuracy/count*
use_locking(
?
"stream_metrics/accuracy/count/readIdentitystream_metrics/accuracy/count*0
_class&
$"loc:@stream_metrics/accuracy/count*
T0*
_output_shapes
: 
_
stream_metrics/accuracy/SizeConst*
_output_shapes
: *
value
B :?*
dtype0
?
stream_metrics/accuracy/ToFloatCaststream_metrics/accuracy/Size*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
g
stream_metrics/accuracy/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
stream_metrics/accuracy/SumSumstream_metrics/ToFloatstream_metrics/accuracy/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
?
!stream_metrics/accuracy/AssignAdd	AssignAddstream_metrics/accuracy/totalstream_metrics/accuracy/Sum*
use_locking( *
_output_shapes
: *0
_class&
$"loc:@stream_metrics/accuracy/total*
T0
?
#stream_metrics/accuracy/AssignAdd_1	AssignAddstream_metrics/accuracy/countstream_metrics/accuracy/ToFloat^stream_metrics/ToFloat*0
_class&
$"loc:@stream_metrics/accuracy/count*
use_locking( *
_output_shapes
: *
T0
f
!stream_metrics/accuracy/Maximum/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
stream_metrics/accuracy/MaximumMaximum"stream_metrics/accuracy/count/read!stream_metrics/accuracy/Maximum/y*
T0*
_output_shapes
: 
?
stream_metrics/accuracy/valueDivNoNan"stream_metrics/accuracy/total/readstream_metrics/accuracy/Maximum*
_output_shapes
: *
T0
h
#stream_metrics/accuracy/Maximum_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
!stream_metrics/accuracy/Maximum_1Maximum#stream_metrics/accuracy/AssignAdd_1#stream_metrics/accuracy/Maximum_1/y*
T0*
_output_shapes
: 
?
!stream_metrics/accuracy/update_opDivNoNan!stream_metrics/accuracy/AssignAdd!stream_metrics/accuracy/Maximum_1*
_output_shapes
: *
T0
r
stream_metrics/precision/CastCastArgMax*
Truncate( *

DstT0
*
_output_shapes	
:?*

SrcT0	
?
stream_metrics/precision/Cast_1Castplaceholders/labels*
Truncate( *#
_output_shapes
:?????????*

SrcT0*

DstT0

q
/stream_metrics/precision/true_positives/Equal/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z
?
-stream_metrics/precision/true_positives/EqualEqualstream_metrics/precision/Cast_1/stream_metrics/precision/true_positives/Equal/y*#
_output_shapes
:?????????*
T0

s
1stream_metrics/precision/true_positives/Equal_1/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
?
/stream_metrics/precision/true_positives/Equal_1Equalstream_metrics/precision/Cast1stream_metrics/precision/true_positives/Equal_1/y*
T0
*
_output_shapes	
:?
?
2stream_metrics/precision/true_positives/LogicalAnd
LogicalAnd-stream_metrics/precision/true_positives/Equal/stream_metrics/precision/true_positives/Equal_1*
_output_shapes	
:?
^
Vstream_metrics/precision/true_positives/assert_type/statically_determined_correct_typeNoOp
?
?stream_metrics/precision/true_positives/count/Initializer/zerosConst*@
_class6
42loc:@stream_metrics/precision/true_positives/count*
dtype0*
valueB
 *    *
_output_shapes
: 
?
-stream_metrics/precision/true_positives/count
VariableV2*
	container *
shared_name *
_output_shapes
: *@
_class6
42loc:@stream_metrics/precision/true_positives/count*
dtype0*
shape: 
?
4stream_metrics/precision/true_positives/count/AssignAssign-stream_metrics/precision/true_positives/count?stream_metrics/precision/true_positives/count/Initializer/zeros*@
_class6
42loc:@stream_metrics/precision/true_positives/count*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
?
2stream_metrics/precision/true_positives/count/readIdentity-stream_metrics/precision/true_positives/count*@
_class6
42loc:@stream_metrics/precision/true_positives/count*
_output_shapes
: *
T0
?
/stream_metrics/precision/true_positives/ToFloatCast2stream_metrics/precision/true_positives/LogicalAnd*

DstT0*
_output_shapes	
:?*

SrcT0
*
Truncate( 
?
0stream_metrics/precision/true_positives/IdentityIdentity2stream_metrics/precision/true_positives/count/read*
T0*
_output_shapes
: 
w
-stream_metrics/precision/true_positives/ConstConst*
_output_shapes
:*
valueB: *
dtype0
?
+stream_metrics/precision/true_positives/SumSum/stream_metrics/precision/true_positives/ToFloat-stream_metrics/precision/true_positives/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
?
1stream_metrics/precision/true_positives/AssignAdd	AssignAdd-stream_metrics/precision/true_positives/count+stream_metrics/precision/true_positives/Sum*
_output_shapes
: *@
_class6
42loc:@stream_metrics/precision/true_positives/count*
use_locking( *
T0
r
0stream_metrics/precision/false_positives/Equal/yConst*
_output_shapes
: *
value	B
 Z *
dtype0

?
.stream_metrics/precision/false_positives/EqualEqualstream_metrics/precision/Cast_10stream_metrics/precision/false_positives/Equal/y*
T0
*#
_output_shapes
:?????????
t
2stream_metrics/precision/false_positives/Equal_1/yConst*
value	B
 Z*
_output_shapes
: *
dtype0

?
0stream_metrics/precision/false_positives/Equal_1Equalstream_metrics/precision/Cast2stream_metrics/precision/false_positives/Equal_1/y*
T0
*
_output_shapes	
:?
?
3stream_metrics/precision/false_positives/LogicalAnd
LogicalAnd.stream_metrics/precision/false_positives/Equal0stream_metrics/precision/false_positives/Equal_1*
_output_shapes	
:?
_
Wstream_metrics/precision/false_positives/assert_type/statically_determined_correct_typeNoOp
?
@stream_metrics/precision/false_positives/count/Initializer/zerosConst*A
_class7
53loc:@stream_metrics/precision/false_positives/count*
dtype0*
_output_shapes
: *
valueB
 *    
?
.stream_metrics/precision/false_positives/count
VariableV2*
shared_name *
_output_shapes
: *
dtype0*
shape: *
	container *A
_class7
53loc:@stream_metrics/precision/false_positives/count
?
5stream_metrics/precision/false_positives/count/AssignAssign.stream_metrics/precision/false_positives/count@stream_metrics/precision/false_positives/count/Initializer/zeros*
use_locking(*A
_class7
53loc:@stream_metrics/precision/false_positives/count*
T0*
validate_shape(*
_output_shapes
: 
?
3stream_metrics/precision/false_positives/count/readIdentity.stream_metrics/precision/false_positives/count*A
_class7
53loc:@stream_metrics/precision/false_positives/count*
_output_shapes
: *
T0
?
0stream_metrics/precision/false_positives/ToFloatCast3stream_metrics/precision/false_positives/LogicalAnd*

SrcT0
*
_output_shapes	
:?*
Truncate( *

DstT0
?
1stream_metrics/precision/false_positives/IdentityIdentity3stream_metrics/precision/false_positives/count/read*
_output_shapes
: *
T0
x
.stream_metrics/precision/false_positives/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
,stream_metrics/precision/false_positives/SumSum0stream_metrics/precision/false_positives/ToFloat.stream_metrics/precision/false_positives/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
?
2stream_metrics/precision/false_positives/AssignAdd	AssignAdd.stream_metrics/precision/false_positives/count,stream_metrics/precision/false_positives/Sum*
use_locking( *
_output_shapes
: *
T0*A
_class7
53loc:@stream_metrics/precision/false_positives/count
?
stream_metrics/precision/addAdd0stream_metrics/precision/true_positives/Identity1stream_metrics/precision/false_positives/Identity*
_output_shapes
: *
T0
g
"stream_metrics/precision/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
 stream_metrics/precision/GreaterGreaterstream_metrics/precision/add"stream_metrics/precision/Greater/y*
T0*
_output_shapes
: 
?
stream_metrics/precision/add_1Add0stream_metrics/precision/true_positives/Identity1stream_metrics/precision/false_positives/Identity*
_output_shapes
: *
T0
?
stream_metrics/precision/divRealDiv0stream_metrics/precision/true_positives/Identitystream_metrics/precision/add_1*
_output_shapes
: *
T0
e
 stream_metrics/precision/value/eConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
stream_metrics/precision/valueSelect stream_metrics/precision/Greaterstream_metrics/precision/div stream_metrics/precision/value/e*
_output_shapes
: *
T0
?
stream_metrics/precision/add_2Add1stream_metrics/precision/true_positives/AssignAdd2stream_metrics/precision/false_positives/AssignAdd*
_output_shapes
: *
T0
i
$stream_metrics/precision/Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
"stream_metrics/precision/Greater_1Greaterstream_metrics/precision/add_2$stream_metrics/precision/Greater_1/y*
_output_shapes
: *
T0
?
stream_metrics/precision/add_3Add1stream_metrics/precision/true_positives/AssignAdd2stream_metrics/precision/false_positives/AssignAdd*
_output_shapes
: *
T0
?
stream_metrics/precision/div_1RealDiv1stream_metrics/precision/true_positives/AssignAddstream_metrics/precision/add_3*
_output_shapes
: *
T0
i
$stream_metrics/precision/update_op/eConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
"stream_metrics/precision/update_opSelect"stream_metrics/precision/Greater_1stream_metrics/precision/div_1$stream_metrics/precision/update_op/e*
_output_shapes
: *
T0
o
stream_metrics/recall/CastCastArgMax*

DstT0
*
Truncate( *

SrcT0	*
_output_shapes	
:?
?
stream_metrics/recall/Cast_1Castplaceholders/labels*

DstT0
*#
_output_shapes
:?????????*

SrcT0*
Truncate( 
n
,stream_metrics/recall/true_positives/Equal/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
?
*stream_metrics/recall/true_positives/EqualEqualstream_metrics/recall/Cast_1,stream_metrics/recall/true_positives/Equal/y*#
_output_shapes
:?????????*
T0

p
.stream_metrics/recall/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
?
,stream_metrics/recall/true_positives/Equal_1Equalstream_metrics/recall/Cast.stream_metrics/recall/true_positives/Equal_1/y*
_output_shapes	
:?*
T0

?
/stream_metrics/recall/true_positives/LogicalAnd
LogicalAnd*stream_metrics/recall/true_positives/Equal,stream_metrics/recall/true_positives/Equal_1*
_output_shapes	
:?
[
Sstream_metrics/recall/true_positives/assert_type/statically_determined_correct_typeNoOp
?
<stream_metrics/recall/true_positives/count/Initializer/zerosConst*=
_class3
1/loc:@stream_metrics/recall/true_positives/count*
valueB
 *    *
_output_shapes
: *
dtype0
?
*stream_metrics/recall/true_positives/count
VariableV2*
_output_shapes
: *
shape: *
	container *=
_class3
1/loc:@stream_metrics/recall/true_positives/count*
dtype0*
shared_name 
?
1stream_metrics/recall/true_positives/count/AssignAssign*stream_metrics/recall/true_positives/count<stream_metrics/recall/true_positives/count/Initializer/zeros*
_output_shapes
: *=
_class3
1/loc:@stream_metrics/recall/true_positives/count*
use_locking(*
validate_shape(*
T0
?
/stream_metrics/recall/true_positives/count/readIdentity*stream_metrics/recall/true_positives/count*
T0*=
_class3
1/loc:@stream_metrics/recall/true_positives/count*
_output_shapes
: 
?
,stream_metrics/recall/true_positives/ToFloatCast/stream_metrics/recall/true_positives/LogicalAnd*

SrcT0
*
Truncate( *

DstT0*
_output_shapes	
:?
?
-stream_metrics/recall/true_positives/IdentityIdentity/stream_metrics/recall/true_positives/count/read*
_output_shapes
: *
T0
t
*stream_metrics/recall/true_positives/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
(stream_metrics/recall/true_positives/SumSum,stream_metrics/recall/true_positives/ToFloat*stream_metrics/recall/true_positives/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
?
.stream_metrics/recall/true_positives/AssignAdd	AssignAdd*stream_metrics/recall/true_positives/count(stream_metrics/recall/true_positives/Sum*=
_class3
1/loc:@stream_metrics/recall/true_positives/count*
use_locking( *
T0*
_output_shapes
: 
o
-stream_metrics/recall/false_negatives/Equal/yConst*
dtype0
*
value	B
 Z*
_output_shapes
: 
?
+stream_metrics/recall/false_negatives/EqualEqualstream_metrics/recall/Cast_1-stream_metrics/recall/false_negatives/Equal/y*
T0
*#
_output_shapes
:?????????
q
/stream_metrics/recall/false_negatives/Equal_1/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
?
-stream_metrics/recall/false_negatives/Equal_1Equalstream_metrics/recall/Cast/stream_metrics/recall/false_negatives/Equal_1/y*
T0
*
_output_shapes	
:?
?
0stream_metrics/recall/false_negatives/LogicalAnd
LogicalAnd+stream_metrics/recall/false_negatives/Equal-stream_metrics/recall/false_negatives/Equal_1*
_output_shapes	
:?
\
Tstream_metrics/recall/false_negatives/assert_type/statically_determined_correct_typeNoOp
?
=stream_metrics/recall/false_negatives/count/Initializer/zerosConst*
dtype0*>
_class4
20loc:@stream_metrics/recall/false_negatives/count*
valueB
 *    *
_output_shapes
: 
?
+stream_metrics/recall/false_negatives/count
VariableV2*>
_class4
20loc:@stream_metrics/recall/false_negatives/count*
dtype0*
shared_name *
_output_shapes
: *
shape: *
	container 
?
2stream_metrics/recall/false_negatives/count/AssignAssign+stream_metrics/recall/false_negatives/count=stream_metrics/recall/false_negatives/count/Initializer/zeros*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*>
_class4
20loc:@stream_metrics/recall/false_negatives/count
?
0stream_metrics/recall/false_negatives/count/readIdentity+stream_metrics/recall/false_negatives/count*
T0*
_output_shapes
: *>
_class4
20loc:@stream_metrics/recall/false_negatives/count
?
-stream_metrics/recall/false_negatives/ToFloatCast0stream_metrics/recall/false_negatives/LogicalAnd*

SrcT0
*
Truncate( *

DstT0*
_output_shapes	
:?
?
.stream_metrics/recall/false_negatives/IdentityIdentity0stream_metrics/recall/false_negatives/count/read*
T0*
_output_shapes
: 
u
+stream_metrics/recall/false_negatives/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
)stream_metrics/recall/false_negatives/SumSum-stream_metrics/recall/false_negatives/ToFloat+stream_metrics/recall/false_negatives/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
?
/stream_metrics/recall/false_negatives/AssignAdd	AssignAdd+stream_metrics/recall/false_negatives/count)stream_metrics/recall/false_negatives/Sum*>
_class4
20loc:@stream_metrics/recall/false_negatives/count*
T0*
use_locking( *
_output_shapes
: 
?
stream_metrics/recall/addAdd-stream_metrics/recall/true_positives/Identity.stream_metrics/recall/false_negatives/Identity*
T0*
_output_shapes
: 
d
stream_metrics/recall/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
stream_metrics/recall/GreaterGreaterstream_metrics/recall/addstream_metrics/recall/Greater/y*
T0*
_output_shapes
: 
?
stream_metrics/recall/add_1Add-stream_metrics/recall/true_positives/Identity.stream_metrics/recall/false_negatives/Identity*
T0*
_output_shapes
: 
?
stream_metrics/recall/divRealDiv-stream_metrics/recall/true_positives/Identitystream_metrics/recall/add_1*
_output_shapes
: *
T0
b
stream_metrics/recall/value/eConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
stream_metrics/recall/valueSelectstream_metrics/recall/Greaterstream_metrics/recall/divstream_metrics/recall/value/e*
T0*
_output_shapes
: 
?
stream_metrics/recall/add_2Add.stream_metrics/recall/true_positives/AssignAdd/stream_metrics/recall/false_negatives/AssignAdd*
_output_shapes
: *
T0
f
!stream_metrics/recall/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
stream_metrics/recall/Greater_1Greaterstream_metrics/recall/add_2!stream_metrics/recall/Greater_1/y*
_output_shapes
: *
T0
?
stream_metrics/recall/add_3Add.stream_metrics/recall/true_positives/AssignAdd/stream_metrics/recall/false_negatives/AssignAdd*
_output_shapes
: *
T0
?
stream_metrics/recall/div_1RealDiv.stream_metrics/recall/true_positives/AssignAddstream_metrics/recall/add_3*
_output_shapes
: *
T0
f
!stream_metrics/recall/update_op/eConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
stream_metrics/recall/update_opSelectstream_metrics/recall/Greater_1stream_metrics/recall/div_1!stream_metrics/recall/update_op/e*
_output_shapes
: *
T0
?
stream_metrics/initNoOp%^stream_metrics/accuracy/count/Assign%^stream_metrics/accuracy/total/Assign!^stream_metrics/mean/count/Assign!^stream_metrics/mean/total/Assign6^stream_metrics/precision/false_positives/count/Assign5^stream_metrics/precision/true_positives/count/Assign3^stream_metrics/recall/false_negatives/count/Assign2^stream_metrics/recall/true_positives/count/Assign"CPT??       ??/?	\?HLB2?AW*?
	
lr??8

e_losses/train????

e_losses/valid???

e_losses/testrB??

e_accuracy/train?:IB

e_accuracy/valid?gB

e_accuracy/test?{B

e_f1_score/train?y?A

e_f1_score/validҤkA

e_f1_score/test4?vA??e?       &]?	??NB2?A?*?
	
lr??8

e_losses/trainK???

e_losses/valid?[??

e_losses/test)???

e_accuracy/train?	ZB

e_accuracy/valid_gB

e_accuracy/test?{B

e_f1_score/train}b|A

e_f1_score/valid??jA

e_f1_score/test4?vAl?]??       &]?	l_QB2?A?*?
	
lr??8

e_losses/train1???

e_losses/valid????

e_losses/test|??

e_accuracy/train\O[B

e_accuracy/valid7?gB

e_accuracy/test?{B

e_f1_score/train?[}A

e_f1_score/valid??mA

e_f1_score/test?M{A2?S"?       &]?	?YTB2?A?*?
	
lr??8

e_losses/trainw???

e_losses/valid????

e_losses/test????

e_accuracy/train0?]B

e_accuracy/valid??lB

e_accuracy/test?]yB

e_f1_score/trainɛ?A

e_f1_score/validY??A

e_f1_score/test9??AF_?a?       &]?	n?VB2?A?*?
	
lr??8

e_losses/train???

e_losses/valid??

e_losses/testX޷?

e_accuracy/train?ccB

e_accuracy/valid?O?B

e_accuracy/test??aB

e_f1_score/train'?A

e_f1_score/valid7?A

e_f1_score/test???A$????       &]?	??2YB2?A?*?
	
lr??8

e_losses/train?/??

e_losses/valid?ݸ?

e_losses/testi???

e_accuracy/train~?kB

e_accuracy/validLĂB

e_accuracy/test?VpB

e_f1_score/train???A

e_f1_score/valid??A

e_f1_score/test䩷A????       &]?	(k?[B2?A?*?
	
lr??8

e_losses/trainື?

e_losses/validZc??

e_losses/test*~??

e_accuracy/trainknB

e_accuracy/valide!?B

e_accuracy/testiB

e_f1_score/train???A

e_f1_score/validS??A

e_f1_score/testJt?Ak#???       &]?	e^B2?A?*?
	
lr??8

e_losses/train?!??

e_losses/valid*ҹ?

e_losses/testy???

e_accuracy/train ?sB

e_accuracy/valid?A?B

e_accuracy/test_?nB

e_f1_score/traina??A

e_f1_score/valid?n?A

e_f1_score/test?A???e?       &]?	??`B2?A?*?
	
lr??8

e_losses/train~???

e_losses/valid???

e_losses/testll??

e_accuracy/train;?yB

e_accuracy/validu?B

e_accuracy/test?v[B

e_f1_score/train???A

e_f1_score/validBW?A

e_f1_score/testٹ?A+T<?       &]?	\
cB2?A?*?
	
lr??8

e_losses/trainC0??

e_losses/validq??

e_losses/test!???

e_accuracy/trainQOyB

e_accuracy/valid???B

e_accuracy/test\MB

e_f1_score/train???A

e_f1_score/valid?Q?A

e_f1_score/test]N?A??h
?       &]?	f"feB2?A?*?
	
lr??8

e_losses/trainƧ?

e_losses/valid.???

e_losses/test R??

e_accuracy/train??|B

e_accuracy/validƀ?B

e_accuracy/test??YB

e_f1_score/trainۗ?A

e_f1_score/valid???A

e_f1_score/test???A??q?       &]?	??gB2?A?*?
	
lr??8

e_losses/train???

e_losses/valid&???

e_losses/test8??

e_accuracy/train??B

e_accuracy/valid???B

e_accuracy/test8XB

e_f1_score/train?B

e_f1_score/valid^R?A

e_f1_score/test???A?r???       &]?	?\jB2?A?*?
	
lr??8

e_losses/train;???

e_losses/valid???

e_losses/testǇ??

e_accuracy/train?B

e_accuracy/valid u?B

e_accuracy/test8.PB

e_f1_score/train?B

e_f1_score/valid???A

e_f1_score/testzD?A??n0?       &]?	0??lB2?A?	*?
	
lr??8

e_losses/train?L??

e_losses/validʓ?

e_losses/test<v??

e_accuracy/train??B

e_accuracy/valid5??B

e_accuracy/test;?wB

e_f1_score/train;?B

e_f1_score/valid҈B

e_f1_score/test|?!BV?q?       &]?	??EoB2?A?
*?
	
lr??8

e_losses/train????

e_losses/valid?B??

e_losses/test?Ԕ?

e_accuracy/train?ډB

e_accuracy/valid?ߖB

e_accuracy/test?Y?B

e_f1_score/train?&B

e_f1_score/valid?a-B

e_f1_score/testJBZA?	?       &]?	# ?qB2?A?
*?
	
lr??8

e_losses/train_???

e_losses/validK???

e_losses/test$$??

e_accuracy/trainJk?B

e_accuracy/valid?	?B

e_accuracy/test??B

e_f1_score/train߭"B

e_f1_score/valid ?1B

e_f1_score/test]?(B????       &]?	*?]tB2?A?*?
	
lr??8

e_losses/trainﴔ?

e_losses/validx??

e_losses/test????

e_accuracy/train?L?B

e_accuracy/valid?]?B

e_accuracy/testU?B

e_f1_score/train??.B

e_f1_score/valid??<B

e_f1_score/test??2B??       &]?	>?vB2?A?*?
	
lr??8

e_losses/train?Ó?

e_losses/valid?"??

e_losses/test????

e_accuracy/train???B

e_accuracy/valid?؟B

e_accuracy/test?+?B

e_f1_score/traind?2B

e_f1_score/valid??=B

e_f1_score/test??'B"(?9?       &]?	QwNyB2?A?*?
	
lr??8

e_losses/train\???

e_losses/valid???

e_losses/test+???

e_accuracy/trainw#?B

e_accuracy/valid???B

e_accuracy/test?B

e_f1_score/train}?8B

e_f1_score/validT8BB

e_f1_score/test??'Bc???       &]?	??{B2?A?*?
	
lr??8

e_losses/train))??

e_losses/validd??

e_losses/test?}t?

e_accuracy/trainǘB

e_accuracy/valid??B

e_accuracy/testEZ?B

e_f1_score/train??;B

e_f1_score/validg?4B

e_f1_score/testH?1B~?Hi?       &]?	}.<~B2?A?*?
	
lr??8

e_losses/train紇?

e_losses/validV?u?

e_losses/test1n??

e_accuracy/trainE??B

e_accuracy/valid??B

e_accuracy/test??B

e_f1_score/trainK?=B

e_f1_score/valid?EB

e_f1_score/testR"B????       &]?	?3ɀB2?A?*?
	
lr??8

e_losses/train?e??

e_losses/validfq?

e_losses/test??

e_accuracy/trainY??B

e_accuracy/valid?X?B

e_accuracy/test??B

e_f1_score/trainJ?DB

e_f1_score/validqBB

e_f1_score/test??&B?g\?       &]?	???B2?A?*?
	
lr??8

e_losses/train; ??

e_losses/validɳ|?

e_losses/test????

e_accuracy/train??B

e_accuracy/valid@ƢB

e_accuracy/test_?nB

e_f1_score/trainY?FB

e_f1_score/validA>CB

e_f1_score/test\?B?????       &]?	m???B2?A?*?
	
lr??8

e_losses/trainn??

e_losses/valid?l?

e_losses/test^'|?

e_accuracy/traine??B

e_accuracy/valid?ʣB

e_accuracy/test???B

e_f1_score/train??EB

e_f1_score/valid0DB

e_f1_score/test?6*BRF?e?       &]?	?y ?B2?A?*?
	
lr??8

e_losses/trainXx?

e_losses/validr?

e_losses/testx???

e_accuracy/train&נB

e_accuracy/valid? ?B

e_accuracy/test?<xB

e_f1_score/train?XSB

e_f1_score/valid^?FB

e_f1_score/test??B՝??       &]?	???B2?A?*?
	
lr??8

e_losses/train)?{?

e_losses/valide?

e_losses/test?9??

e_accuracy/trainǓ?B

e_accuracy/valid?5?B

e_accuracy/test1%?B

e_f1_score/train??SB

e_f1_score/valid?\HB

e_f1_score/test?-0B??l?       &]?		??B2?A?*?
	
lr??8

e_losses/train?vv?

e_losses/valid?p?

e_losses/test????

e_accuracy/train?u?B

e_accuracy/valid`??B

e_accuracy/testU?B

e_f1_score/train`ZXB

e_f1_score/valid?BHB

e_f1_score/test?B?o???       &]?	??`?B2?A?*?
	
lr??8

e_losses/train?k?

e_losses/validS?a?

e_losses/test?s??

e_accuracy/train;ơB

e_accuracy/valid_??B

e_accuracy/test?m?B

e_f1_score/trainz?]B

e_f1_score/validNB

e_f1_score/testty.B>? ??       &]?	1&??B2?A?*?
	
lr??8

e_losses/traind?k?

e_losses/validf?h?

e_losses/test2|?

e_accuracy/train???B

e_accuracy/validy??B

e_accuracy/test???B

e_f1_score/train??]B

e_f1_score/valid?vGB

e_f1_score/test?CQB8?β?       &]?	????B2?A?*?
	
lr??8

e_losses/train??m?

e_losses/valid?
f?

e_losses/test?q??

e_accuracy/train??B

e_accuracy/valid???B

e_accuracy/test?Y?B

e_f1_score/train?pYB

e_f1_score/validM?FB

e_f1_score/test?%B????       &]?	???B2?A?*?
	
lr??8

e_losses/trainU8a?

e_losses/valid??_?

e_losses/test{?

e_accuracy/train???B

e_accuracy/validh ?B

e_accuracy/test???B

e_f1_score/train?bB

e_f1_score/validGjSB

e_f1_score/testg?AB?<k??       &]?	?7?B2?A?*?
	
lr??8

e_losses/train?W?

e_losses/valid?0b?

e_losses/test?%X?

e_accuracy/train?U?B

e_accuracy/validw?B

e_accuracy/test?F?B

e_f1_score/train?nB

e_f1_score/valid??VB

e_f1_score/test? PB:????       &]?	>??B2?A?*?
	
lr??8

e_losses/train??X?

e_losses/valid/?i?

e_losses/test??L?

e_accuracy/train?ФB

e_accuracy/valid??B

e_accuracy/testX??B

e_f1_score/trainWkB

e_f1_score/valid??WB

e_f1_score/test??eB7???       &]?	K?B2?A?*?
	
lr??8

e_losses/trainf?_?

e_losses/validR?

e_losses/test?ji?

e_accuracy/train돣B

e_accuracy/validwݧB

e_accuracy/test?B

e_f1_score/trainFJnB

e_f1_score/validrSaB

e_f1_score/test?oB?Y?       &]?	TŜ?B2?A?*?
	
lr??8

e_losses/trainݠQ?

e_losses/valid/'_?

e_losses/testN!}?

e_accuracy/train?ǧB

e_accuracy/valid???B

e_accuracy/test2??B

e_f1_score/train?]}B

e_f1_score/validAIWB

e_f1_score/test??\B׉??       &]?	F?B2?A?*?
	
lr??8

e_losses/train3nJ?

e_losses/valid?oP?

e_losses/test??X?

e_accuracy/train
I?B

e_accuracy/valid%[?B

e_accuracy/test?t?B

e_f1_score/train=kzB

e_f1_score/valid??cB

e_f1_score/test~P}B???k?       &]?	m?d?B2?A?*?
	
lr??8

e_losses/trainyTF?

e_losses/validP?_?

e_losses/test?7R?

e_accuracy/trainF	?B

e_accuracy/validx??B

e_accuracy/test3לB

e_f1_score/train~??B

e_f1_score/validnSB

e_f1_score/test"moB????       &]?	?B2?A?*?
	
lr??8

e_losses/train?D??

e_losses/valid ?J?

e_losses/test?=?

e_accuracy/train?$?B

e_accuracy/validnd?B

e_accuracy/test?M?B

e_f1_score/train?a?B

e_f1_score/validsBnB

e_f1_score/teste?BU????       &]?	A?ުB2?A?*?
	
lr??8

e_losses/train^?<?

e_losses/valid??J?

e_losses/test?r0?

e_accuracy/trainQ?B

e_accuracy/valid$T?B

e_accuracy/testѩB

e_f1_score/train(>?B

e_f1_score/valid??sB

e_f1_score/testBH?B?<???       &]?	=?y?B2?A?*?
	
lr??8

e_losses/trainn?<?

e_losses/valid?-H?

e_losses/test??1?

e_accuracy/train???B

e_accuracy/valid???B

e_accuracy/test4??B

e_f1_score/train???B

e_f1_score/valid??yB

e_f1_score/test+?B?ď?       &]?	??"?B2?A?*?
	
lr??8

e_losses/trainn7?

e_losses/valid??9?

e_losses/test&BD?

e_accuracy/train":?B

e_accuracy/valid?ʪB

e_accuracy/test???B

e_f1_score/trainJ6?B

e_f1_score/validZ??B

e_f1_score/test??B'?{d?       &]?	Ⱥ??B2?A?*?
	
lr??8

e_losses/trainT5?

e_losses/validqS?

e_losses/test?u-?

e_accuracy/train?f?B

e_accuracy/validO??B

e_accuracy/test?@?B

e_f1_score/train?8?B

e_f1_score/valid2?qB

e_f1_score/testx?B?&^N?       &]?	F}??B2?A?*?
	
lr??8

e_losses/train??3?

e_losses/validNv8?

e_losses/test??3?

e_accuracy/train?~?B

e_accuracy/valid}A?B

e_accuracy/test4??B

e_f1_score/train???B

e_f1_score/validxςB

e_f1_score/test?BA???       &]?	??I?B2?A?*?
	
lr??8

e_losses/train?3?

e_losses/valid?+?

e_losses/testWe7?

e_accuracy/trainu?B

e_accuracy/validsȭB

e_accuracy/testk?B

e_f1_score/train?F?B

e_f1_score/valid?W?B

e_f1_score/testi?B?~??       &]?	??B2?A?*?
	
lr??8

e_losses/train?-?

e_losses/valid?/?

e_losses/testT?4?

e_accuracy/train???B

e_accuracy/valid?5?B

e_accuracy/test?,?B

e_f1_score/train?-?B

e_f1_score/valid?I?B

e_f1_score/test@r?Bz?N?       &]?		???B2?A?*?
	
lr??8

e_losses/train?o&?

e_losses/valid??N?

e_losses/test+0?

e_accuracy/train姭B

e_accuracy/valid???B

e_accuracy/test?g?B

e_f1_score/train?ґB

e_f1_score/valid??pB

e_f1_score/test#?B?[ʭ?       &]?	}?K?B2?A?*?
	
lr??8

e_losses/train??&?

e_losses/valid_E?

e_losses/testu??

e_accuracy/train?A?B

e_accuracy/validᵩB

e_accuracy/test4??B

e_f1_score/train³?B

e_f1_score/valid?~B

e_f1_score/test)??B??X??       &]?	?[??B2?A? *?
	
lr??8

e_losses/train??&?

e_losses/validd?*?

e_losses/test??!?

e_accuracy/trainfx?B

e_accuracy/valid!F?B

e_accuracy/test???B

e_f1_score/train?t?B

e_f1_score/validN?B

e_f1_score/test???Bl??       &]?	,h??B2?A?!*?
	
lr??8

e_losses/trainӈ?

e_losses/valid?&?

e_losses/testzc?

e_accuracy/train?w?B

e_accuracy/valid?.?B

e_accuracy/test":?B

e_f1_score/traing?B

e_f1_score/valid?}?B

e_f1_score/test?+?B3KN??       &]?	?MQ?B2?A?!*?
	
lr??8

e_losses/train???

e_losses/valid??1?

e_losses/test??"?

e_accuracy/train?#?B

e_accuracy/valid*??B

e_accuracy/test???B

e_f1_score/train*I?B

e_f1_score/valid??B

e_f1_score/testt??B???D?       &]?	?p?B2?A?"*?
	
lr??8

e_losses/train?_?

e_losses/valid?&6?

e_losses/testY%?

e_accuracy/train9Y?B

e_accuracy/valide??B

e_accuracy/testF?B

e_f1_score/trainD?B

e_f1_score/validBX?B

e_f1_score/testci?B?l)"?       &]?	Kg??B2?A?#*?
	
lr??8

e_losses/train$??

e_losses/valid??$?

e_losses/testc??

e_accuracy/train?b?B

e_accuracy/valid(??B

e_accuracy/test?3?B

e_f1_score/train_J?B

e_f1_score/valid?*?B

e_f1_score/test??B??.??       &]?	v???B2?A?#*?
	
lr??8

e_losses/train?1?

e_losses/valid[u?

e_losses/test?X?

e_accuracy/train??B

e_accuracy/valid? ?B

e_accuracy/test4??B

e_f1_score/trainE~?B

e_f1_score/valid??B

e_f1_score/test?ʆB??3??       &]?	v`T?B2?A?$*?
	
lr??8

e_losses/train??

e_losses/validf??

e_losses/testD?%?

e_accuracy/trainY??B

e_accuracy/valid8??B

e_accuracy/test?,?B

e_f1_score/trainx??B

e_f1_score/valid???B

e_f1_score/test]??B?B???       &]?	????B2?A?%*?
	
lr??8

e_losses/train?-?

e_losses/valid??

e_losses/testd??

e_accuracy/trainT??B

e_accuracy/valid9??B

e_accuracy/test}a?B

e_f1_score/train;ѝB

e_f1_score/valid?r?B

e_f1_score/test?>?B?o???       &]?	???B2?A?&*?
	
lr??8

e_losses/train??

e_losses/validt?#?

e_losses/test?? ?

e_accuracy/trainD4?B

e_accuracy/valid w?B

e_accuracy/test???B

e_f1_score/train??B

e_f1_score/validɌB

e_f1_score/test1?BRym?       &]?	+?_?B2?A?&*?
	
lr??8

e_losses/train)?

e_losses/valid)??

e_losses/test??4?

e_accuracy/trainBȱB

e_accuracy/validB

e_accuracy/test!a?B

e_f1_score/train?B

e_f1_score/valid???B

e_f1_score/test??wB_???       &]?	?v??B2?A?'*?
	
lr??8

e_losses/train7
?

e_losses/valid?k?

e_losses/test*??

e_accuracy/train?۰B

e_accuracy/valid???B

e_accuracy/test5??B

e_f1_score/trainmN?B

e_f1_score/valid
??B

e_f1_score/testN{?Bݝɣ?       &]?	G@??B2?A?(*?
	
lr??8

e_losses/train???

e_losses/valid?<?

e_losses/test???

e_accuracy/train???B

e_accuracy/valid8??B

e_accuracy/test?{?B

e_f1_score/trainש?B

e_f1_score/valid6?B

e_f1_score/test@;?B???p?       &]?	?W?B2?A?(*?
	
lr??8

e_losses/trainD??

e_losses/validi??

e_losses/test??

e_accuracy/train1?B

e_accuracy/valid? ?B

e_accuracy/testG??B

e_f1_score/train)??B

e_f1_score/valid??B

e_f1_score/test??B:?-`?       &]?	J???B2?A?)*?
	
lr??8

e_losses/train
??

e_losses/valid?%?

e_losses/testR??

e_accuracy/trainEM?B

e_accuracy/validˮ?B

e_accuracy/testYu?B

e_f1_score/train???B

e_f1_score/valid~âB

e_f1_score/testՙ?Bu"??       &]?	?e??B2?A?**?
	
lr??8

e_losses/trainD??

e_losses/valid??

e_losses/test?E?>

e_accuracy/train?G?B

e_accuracy/validM?B

e_accuracy/test?&?B

e_f1_score/trainOC?B

e_f1_score/valid4??B

e_f1_score/test?Q?B`<?s?       &]?	?_?B2?A?**?
	
lr??8

e_losses/train?? ?

e_losses/valid?"?

e_losses/test?7?>

e_accuracy/train|j?B

e_accuracy/valid?}?B

e_accuracy/test# ?B

e_f1_score/train?K?B

e_f1_score/valid???B

e_f1_score/test?u?B|u???       &]?	`???B2?A?+*?
	
lr??8

e_losses/train???>

e_losses/valid?
?

e_losses/test???>

e_accuracy/trainh??B

e_accuracy/valid?m?B

e_accuracy/test5??B

e_f1_score/train֟?B

e_f1_score/validp?B

e_f1_score/test???B?q???       &]?	????B2?A?,*?
	
lr??8

e_losses/train۬?>

e_losses/valid??

e_losses/test?r?

e_accuracy/train??B

e_accuracy/valid? ?B

e_accuracy/testG?B

e_f1_score/traincC?B

e_f1_score/valid?M?B

e_f1_score/test?"?B????       &]?	??D?B2?A?,*?
	
lr??8

e_losses/train?"?>

e_losses/validBj?

e_losses/test?j?>

e_accuracy/train&??B

e_accuracy/valid???B

e_accuracy/test???B

e_f1_score/train7?B

e_f1_score/valid?B

e_f1_score/test???BG???       &]?	???B2?A?-*?
	
lr??8

e_losses/train?i?>

e_losses/valid'??

e_losses/test??>

e_accuracy/traing?B

e_accuracy/valid? ?B

e_accuracy/test???B

e_f1_score/train?T?B

e_f1_score/valid46?B

e_f1_score/test??B ??p?       &]?	\\?B2?A?.*?
	
lr??8

e_losses/train??>

e_losses/valid`??

e_losses/test]?

e_accuracy/trainٳB

e_accuracy/validO??B

e_accuracy/test???B

e_f1_score/train???B

e_f1_score/valid?=?B

e_f1_score/test?:?Bs????       &]?	2:]?B2?A?.*?
	
lr??8

e_losses/train???>

e_losses/valid<? ?

e_losses/test?*?

e_accuracy/trainu?B

e_accuracy/valid???B

e_accuracy/test?{?B

e_f1_score/trainmͣB

e_f1_score/valid6ӦB

e_f1_score/testV?B???;?       &]?	??B2?A?/*?
	
lr??8

e_losses/train}(?>

e_losses/valid??>

e_losses/test???>

e_accuracy/train[??B

e_accuracy/validF?B

e_accuracy/test?n?B

e_f1_score/train?åB

e_f1_score/valid??B

e_f1_score/test6??B?3X?       &]?	??? C2?A?0*?
	
lr??8

e_losses/train???>

e_losses/valid*??>

e_losses/testi%?

e_accuracy/trains??B

e_accuracy/valid??B

e_accuracy/test~T?B

e_f1_score/trainbΦB

e_f1_score/valid??B

e_f1_score/test?r?BmRF?       &]?	?ZtC2?A?0*?
	
lr??8

e_losses/train+}?>

e_losses/valid???>

e_losses/testT?
?

e_accuracy/trainyR?B

e_accuracy/valid?óB

e_accuracy/test???B

e_f1_score/traind??B

e_f1_score/valid?§B

e_f1_score/testYֆBy?(?       &]?	_?8C2?A?1*?
	
lr??8

e_losses/train??>

e_losses/valid???>

e_losses/test???>

e_accuracy/train?R?B

e_accuracy/valid=??B

e_accuracy/test?{?B

e_f1_score/train???B

e_f1_score/validlx?B

e_f1_score/test?΋BA????       &]?	???C2?A?2*?
	
lr??8

e_losses/train???>

e_losses/valid???>

e_losses/test???>

e_accuracy/trainj?B

e_accuracy/valid??B

e_accuracy/test?n?B

e_f1_score/trainl??B

e_f1_score/validB&?B

e_f1_score/test??B?A;??       &]?	ѧ?C2?A?2*?
	
lr??8

e_losses/train?{?>

e_losses/validl}?>

e_losses/test?:?>

e_accuracy/train???B

e_accuracy/valid^ȴB

e_accuracy/test?ײB

e_f1_score/train.ͦB

e_f1_score/valid?ߦB

e_f1_score/test??Bվ?       &]?	??C2?A?3*?
	
lr??8

e_losses/train?)?>

e_losses/valid??>

e_losses/test??>

e_accuracy/train???B

e_accuracy/valid?J?B

e_accuracy/test?&?B

e_f1_score/trainT?B

e_f1_score/valid???B

e_f1_score/test[??B?C0??       &]?	]/bC2?A?4*?
	
lr??8

e_losses/train???>

e_losses/valid//?>

e_losses/test??>

e_accuracy/train$?B

e_accuracy/valid? ?B

e_accuracy/testG?B

e_f1_score/trains^?B

e_f1_score/valid?
?B

e_f1_score/testM?B@?6??       &]?	??C2?A?4*?
	
lr??8

e_losses/trainE??>

e_losses/valid&U?>

e_losses/testr;?>

e_accuracy/train???B

e_accuracy/validf:?B

e_accuracy/test5??B

e_f1_score/train???B

e_f1_score/valid??B

e_f1_score/test???B?C?b?       &]?	???C2?A?5*?
	
lr??8

e_losses/trainhZ?>

e_losses/valid???>

e_losses/test?S?

e_accuracy/train???B

e_accuracy/valid?óB

e_accuracy/test???B

e_f1_score/train[??B

e_f1_score/valid??B

e_f1_score/test???Bti??       &]?	2?C2?A?6*?
	
lr??8

e_losses/train???>

e_losses/valid???>

e_losses/test\??>

e_accuracy/train"??B

e_accuracy/valid?J?B

e_accuracy/test???B

e_f1_score/trainhŧB

e_f1_score/valid???B

e_f1_score/test&?B:/?       &]?	??iC2?A?6*?
	
lr??8

e_losses/train??>

e_losses/valid?P?>

e_losses/test[]?>

e_accuracy/train?B

e_accuracy/validEk?B

e_accuracy/test???B

e_f1_score/trainyߧB

e_f1_score/validX?B

e_f1_score/test-??B80?Q?       &]?	??C2?A?7*?
	
lr??8

e_losses/train???>

e_losses/valid?k?

e_losses/testT??>

e_accuracy/trainQ??B

e_accuracy/valid?Q?B

e_accuracy/test???B

e_f1_score/train???B

e_f1_score/valid???B

e_f1_score/test???B???       &]?	D?!C2?A?8*?
	
lr??8

e_losses/train?V?>

e_losses/valid?|?>

e_losses/test+}?>

e_accuracy/train?ȵB

e_accuracy/validf:?B

e_accuracy/test?M?B

e_f1_score/train???B

e_f1_score/valid?S?B

e_f1_score/test?M?B?(?       &]?	??~$C2?A?9*?
	
lr??8

e_losses/train;??>

e_losses/valid T?>

e_losses/test??>

e_accuracy/train#??B

e_accuracy/valid??B

e_accuracy/test4??B

e_f1_score/train??B

e_f1_score/validǰ?B

e_f1_score/test???Bo??5?       &]?	3<'C2?A?9*?
	
lr??8

e_losses/train1??>

e_losses/valid*?>

e_losses/test??

e_accuracy/train???B

e_accuracy/validF?B

e_accuracy/testX??B

e_f1_score/trainD??B

e_f1_score/valid???B

e_f1_score/testK??B??Ѵ?       &]?	>??)C2?A?:*?
	
lr??8

e_losses/traina??>

e_losses/validzu?>

e_losses/test???>

e_accuracy/traing??B

e_accuracy/valid??B

e_accuracy/test???B

e_f1_score/train??B

e_f1_score/valid?g?B

e_f1_score/test]|?B??fM?       &]?	ߒ,C2?A?;*?
	
lr??8

e_losses/trainR<?>

e_losses/valid?~?>

e_losses/testQ	?

e_accuracy/train+??B

e_accuracy/valid?<?B

e_accuracy/test}n?B

e_f1_score/trainƪ?B

e_f1_score/validgB?B

e_f1_score/test?r?B?څ?       &]?	?:/C2?A?;*?
	
lr??8

e_losses/trainA??>

e_losses/validiV?>

e_losses/test?3?>

e_accuracy/train???B

e_accuracy/valid=??B

e_accuracy/test???B

e_f1_score/train?m?B

e_f1_score/validɒ?B

e_f1_score/test??Bu????       &]?	???1C2?A?<*?
	
lr??8

e_losses/train?7?>

e_losses/valid6??>

e_losses/test4??>

e_accuracy/train
??B

e_accuracy/valid?óB

e_accuracy/test?@?B

e_f1_score/train:??B

e_f1_score/validșB

e_f1_score/teste?Bᵼ?       &]?	:?4C2?A?=*?
	
lr??8

e_losses/trainh??>

e_losses/valid???>

e_losses/test???>

e_accuracy/train?F?B

e_accuracy/valid?Q?B

e_accuracy/test4??B

e_f1_score/train2??B

e_f1_score/valid夤B

e_f1_score/testKJ?Bj??=?       &]?	???7C2?A?=*?
	
lr??8

e_losses/trainnO?>

e_losses/valid??>

e_losses/testD??

e_accuracy/train???B

e_accuracy/valid??B

e_accuracy/testF?B

e_f1_score/train??B

e_f1_score/valid???B

e_f1_score/test?ŎB??	?       &]?	Gv?9C2?A?>*?
	
lr??8

e_losses/train?!?>

e_losses/valid?Q ?

e_losses/test?a?>

e_accuracy/train???B

e_accuracy/validGr?B

e_accuracy/test???B

e_f1_score/train?ܪB

e_f1_score/validR??B

e_f1_score/test\/?Bus?t?       &]?	??<C2?A??*?
	
lr??8

e_losses/train?ÿ>

e_losses/valid(?>

e_losses/test?q?>

e_accuracy/trainGZ?B

e_accuracy/valid???B

e_accuracy/test"-?B

e_f1_score/train??B

e_f1_score/valid?7?B

e_f1_score/test??B??]$?       &]?	?dQ?C2?A??*?
	
lr??8

e_losses/train??>

e_losses/valid?_?

e_losses/testiK?

e_accuracy/train???B

e_accuracy/valid?'?B

e_accuracy/test!T?B

e_f1_score/train΄?B

e_f1_score/valid?d?B

e_f1_score/test???B????       &]?	ƣ?AC2?A?@*?
	
lr??8

e_losses/train`?>

e_losses/valid)??>

e_losses/test9c?

e_accuracy/trainp??B

e_accuracy/valid/?B

e_accuracy/test?Z?B

e_f1_score/train?z?B

e_f1_score/valid??B

e_f1_score/testb?B?\%?       &]?	ב?DC2?A?A*?
	
lr??8

e_losses/train|?>

e_losses/valid9??>

e_losses/test{n?>

e_accuracy/train???B

e_accuracy/validڋ?B

e_accuracy/testY??B

e_f1_score/train?P?B

e_f1_score/valid??B

e_f1_score/test?%?B?k<[?       &]?	??^GC2?A?A*?
	
lr??8

e_losses/train???>

e_losses/valid???>

e_losses/test???>

e_accuracy/trainQ??B

e_accuracy/valid?a?B

e_accuracy/testG??B

e_f1_score/train?B

e_f1_score/valid#ǦB

e_f1_score/test??B???>?       &]?	???IC2?A?B*?
	
lr??8

e_losses/train???>

e_losses/valid???>

e_losses/test???>

e_accuracy/train???B

e_accuracy/valid-?B

e_accuracy/test???B

e_f1_score/train??B

e_f1_score/valid#I?B

e_f1_score/testB??D??       &]?	??LC2?A?C*?
	
lr??8

e_losses/train?O?>

e_losses/valid0??>

e_losses/testY??>

e_accuracy/train?-?B

e_accuracy/validO??B

e_accuracy/test???B

e_f1_score/train\=?B

e_f1_score/valid???B

e_f1_score/testSv?B1]H?       &]?	??OOC2?A?C*?
	
lr??8

e_losses/traini	?>

e_losses/validjC?>

e_losses/test?r?>

e_accuracy/train?f?B

e_accuracy/valid??B

e_accuracy/testYu?B

e_f1_score/train???B

e_f1_score/valid??B

e_f1_score/test?#?B??ɰ?       &]?	???QC2?A?D*?
	
lr??8

e_losses/train???>

e_losses/valid??>

e_losses/testu?>

e_accuracy/train??B

e_accuracy/valid?óB

e_accuracy/test~G?B

e_f1_score/trainO@?B

e_f1_score/validߠ?B

e_f1_score/testgĝB~?}??       &]?	?[?TC2?A?E*?
	
lr??8

e_losses/train]?>

e_losses/valid}??>

e_losses/test\K?

e_accuracy/trainb??B

e_accuracy/valid???B

e_accuracy/test??B

e_f1_score/train?ثB

e_f1_score/valid*?B

e_f1_score/test?2?B?;???       &]?	?)<WC2?A?E*?
	
lr??8

e_losses/train???>

e_losses/valid? ?>

e_losses/test?c?>

e_accuracy/train?^?B

e_accuracy/valid?óB

e_accuracy/test???B

e_f1_score/train8׫B

e_f1_score/valid?¦B

e_f1_score/testAb?BZ
X?       &]?	??YC2?A?F*?
	
lr??8

e_losses/train?o?>

e_losses/valid???>

e_losses/test???>

e_accuracy/train`??B

e_accuracy/valid???B

e_accuracy/test??B

e_f1_score/train`F?B

e_f1_score/validv??B

e_f1_score/testE7?B?a???       &]?	??\C2?A?G*?
	
lr??8

e_losses/train6d?>

e_losses/valid|??>

e_losses/test>?>

e_accuracy/train???B

e_accuracy/validF?B

e_accuracy/test???B

e_f1_score/train?H?B

e_f1_score/validݑ?B

e_f1_score/test?F?B?g?%?       &]?	k?"_C2?A?G*?
	
lr??8

e_losses/train??>

e_losses/valid?w?>

e_losses/test?J?>

e_accuracy/train?a?B

e_accuracy/valid? ?B

e_accuracy/testlްB

e_f1_score/train߷?B

e_f1_score/validGĤB

e_f1_score/testB?Bc?cD?       &]?	7??aC2?A?H*?
	
lr??8

e_losses/trainW??>

e_losses/validƍ?>

e_losses/test~?>

e_accuracy/trainB

e_accuracy/validԳB

e_accuracy/testYu?B

e_f1_score/train?K?B

e_f1_score/valid?a?B

e_f1_score/test???Bݎ??       &]?	???dC2?A?I*?
	
lr??8

e_losses/train???>

e_losses/valid?S?>

e_losses/testj+?

e_accuracy/trainI?B

e_accuracy/valid^ȴB

e_accuracy/testY??B

e_f1_score/train???B

e_f1_score/valid???B

e_f1_score/test??B,???       &]?	?O1gC2?A?I*?
	
lr??8

e_losses/train@??>

e_losses/valid???>

e_losses/test???>

e_accuracy/train?J?B

e_accuracy/validʧ?B

e_accuracy/test??B

e_f1_score/train?*?B

e_f1_score/valid???B

e_f1_score/test˟BcQ???       &]?	?iC2?A?J*?
	
lr??8

e_losses/train`t?>

e_losses/valid?Q?>

e_losses/test?Y?>

e_accuracy/train?}?B

e_accuracy/valid?óB

e_accuracy/test???B

e_f1_score/train*??B

e_f1_score/valid?B?B

e_f1_score/testx??Bc<???       &]?	V??lC2?A?K*?
	
lr??8

e_losses/trainO?>

e_losses/validr?>

e_losses/test???>

e_accuracy/train?^?B

e_accuracy/valid?J?B

e_accuracy/testYu?B

e_f1_score/train+??B

e_f1_score/valid?ɧB

e_f1_score/test?}?B)ݏ??       &]?	??;oC2?A?L*?
	
lr??8

e_losses/train??>

e_losses/valid??>

e_losses/testq-?

e_accuracy/train?G?B

e_accuracy/valid?a?B

e_accuracy/test?g?B

e_f1_score/trainCB?B

e_f1_score/valid7ңB

e_f1_score/testm??B?:|??       &]?	i?qC2?A?L*?
	
lr??8

e_losses/trainD?>

e_losses/valid`_?>

e_losses/test?!?>

e_accuracy/train1?B

e_accuracy/valid??B

e_accuracy/test???B

e_f1_score/train&"?B

e_f1_score/valid???B

e_f1_score/testD͕BJ??1?       &]?	?}?tC2?A?M*?
	
lr??8

e_losses/train???>

e_losses/valid e?>

e_losses/test???>

e_accuracy/trainNX?B

e_accuracy/valid?óB

e_accuracy/testѩB

e_f1_score/train???B

e_f1_score/validrХB

e_f1_score/testW?B?0T??       &]?	wC2?A?N*?
	
lr??8

e_losses/trainҵ?>

e_losses/valid?C?>

e_losses/test??>

e_accuracy/train??B

e_accuracy/valid-?B

e_accuracy/test?{?B

e_f1_score/trainB??B

e_f1_score/validh9?B

e_f1_score/testH??B?AM-?       &]?	gx?yC2?A?N*?
	
lr??8

e_losses/train?L?>

e_losses/valid??>

e_losses/test+g?>

e_accuracy/trainZ?B

e_accuracy/valid?Q?B

e_accuracy/test# ?B

e_f1_score/train?O?B

e_f1_score/valid???B

e_f1_score/test
1?Bٽ?w?       &]?	1~{|C2?A?O*?
	
lr??8

e_losses/trainh??>

e_losses/validD??>

e_losses/test?B?

e_accuracy/train?~?B

e_accuracy/valid???B

e_accuracy/testF&?B

e_f1_score/train6??B

e_f1_score/valid??B

e_f1_score/test?&?B?z???       &]?	$?C2?A?P*?
	
lr??8

e_losses/train)r?>

e_losses/valid6?>

e_losses/test??>

e_accuracy/trainU??B

e_accuracy/valid??B

e_accuracy/test?g?B

e_f1_score/traingN?B

e_f1_score/valid?B

e_f1_score/testOl?B????       &]?	????C2?A?P*?
	
lr??8

e_losses/train쉛>

e_losses/valid???>

e_losses/test.#?>

e_accuracy/train??B

e_accuracy/valid?5?B

e_accuracy/test??B

e_f1_score/trainz?B

e_f1_score/validꅦB

e_f1_score/test?-?B???C?       &]?	8nt?C2?A?Q*?
	
lr??8

e_losses/train??>

e_losses/validI??>

e_losses/test?E?>

e_accuracy/trainiG?B

e_accuracy/valid$??B

e_accuracy/test??B

e_f1_score/train1=?B

e_f1_score/valid?ҧB

e_f1_score/test??B{??v?       &]?	???C2?A?R*?
	
lr??8

e_losses/train???>

e_losses/validT??>

e_losses/test???

e_accuracy/train???B

e_accuracy/valid*?B

e_accuracy/test?3?B

e_f1_score/train?կB

e_f1_score/valid?Z?B

e_f1_score/test'H?B?@???       &]?	tݛ?C2?A?R*?
	
lr??8

e_losses/traink??>

e_losses/valid???>

e_losses/testw??>

e_accuracy/train?p?B

e_accuracy/valid?óB

e_accuracy/test?M?B

e_f1_score/train0
?B

e_f1_score/valid???B

e_f1_score/test??BV?F??       &]?	&c?C2?A?S*?
	
lr??8

e_losses/train?G?>

e_losses/valid?j?>

e_losses/testb	?>

e_accuracy/trainP??B

e_accuracy/valid5??B

e_accuracy/test"-?B

e_f1_score/trainʴ?B

e_f1_score/valid2N?B

e_f1_score/test??B?tK??       &]?	;??C2?A?T*?
	
lr??8

e_losses/train'[?>

e_losses/validsO?>

e_losses/testqA?>

e_accuracy/trainw?B

e_accuracy/valid???B

e_accuracy/test?M?B

e_f1_score/train=?B

e_f1_score/valid?\?B

e_f1_score/test??B??S?       &]?	????C2?A?T*?
	
lr??8

e_losses/train??>

e_losses/valid&??>

e_losses/test??>

e_accuracy/train=¸B

e_accuracy/valid?óB

e_accuracy/test?{?B

e_f1_score/train???B

e_f1_score/valid??B

e_f1_score/testȰ?B????       &]?	?$\?C2?A?U*?
	
lr??8

e_losses/train?c?>

e_losses/valid???>

e_losses/testD.?>

e_accuracy/train???B

e_accuracy/valid?h?B

e_accuracy/test?M?B

e_f1_score/train?)?B

e_f1_score/valid;ǢB

e_f1_score/test???B?ⷝ?       &]?	[???C2?A?V*?
	
lr??8

e_losses/train??>

e_losses/valid?|?>

e_losses/test?V?>

e_accuracy/trainUU?B

e_accuracy/valid???B

e_accuracy/test5??B

e_f1_score/trainyw?B

e_f1_score/valid???B

e_f1_score/test?ߜB9B`