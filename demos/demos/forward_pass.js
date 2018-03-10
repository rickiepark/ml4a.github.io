var demo = function(parent, width, height, datasetName_, snapshot_, testAll_, numTrain_, numTest_) 
{	
	// setup canvas
	var canvas = parent.canvas;
	var ctx = canvas.getContext('2d');

	var datasetName = (datasetName_ === undefined) ? 'MNIST' : datasetName_;
	var snapshot = (snapshot_ === undefined) ? '/demos/datasets/mnist/mnist_snapshot_2layers.json' : snapshot_;
	var numTrain = (numTrain_ === undefined) ? 30000 : numTrain_;
	var numTest = (numTest_ === undefined) ? 500 : numTest_;

	var space_between = 100;
	var output_neuron_radius = 14;
	var bar_height = 10;
	var max_bar_width = 100;
	var label_font_size = 18;
	var sample_scale = 4.0;

	// variables
	var data, net, classes, nc, dim;
	var vis, vis_settings, max_label_width, x1, y1, x2, y2, xm, ym;
	var idx = 2;

	function preloadModel(datasetName_, snapshot, callback) {
		datasetName = datasetName_;
	    data = new dataset(datasetName);
	    net = new convnet(data);
	    classes = data.get_classes();
	    nc = classes.length;
	    dim = data.get_dim();
	    net.load_from_json(snapshot, callback);
	};

	function createModel(callback) {
	    data = new dataset(datasetName);
	    net = new convnet(data);
	    net.add_layer({type:'fc', num_neurons:10, activation:'sigmoid'});
	    net.add_layer({type:'softmax', num_classes:10});
	    net.setup_trainer({method:'adadelta', learning_rate:0.1, batch_size:8, l2_decay:0.0001});
	    classes = data.get_classes();
	    nc = classes.length;
	    dim = data.get_dim();
	    callback();
	};

	function test_all() {
	    net.test(0, numTest, update_canvas);
	};

	function test_next_sample_auto(result) {
	    update_canvas(result);
	    setTimeout(function() {
	    	idx += 1;
	        net.test(idx, idx+1, test_next_sample_auto);   // when to stop?
	    }, 3000);
	};

	function bezier(ctx, x1, y1, x2, y2, x3, y3, x4, y4) {
	    ctx.beginPath();
	    ctx.lineWidth = 3.0;
	    ctx.moveTo(x1,y1);
	    ctx.bezierCurveTo(
	        x2,y2,
	        x3,y3,
	        x4,y4);
	    ctx.stroke();
	    ctx.closePath();
	};

	function update_canvas(result) {
	    ctx.save();
	    ctx.fillStyle = 'rgba(255,255,255,1.0)';
	    ctx.fillRect(0, 0, canvas.width, canvas.height);

	    // draw sample
	    data.draw_sample(ctx, idx+1, 5, ym-data.get_dim()*sample_scale/2.0, sample_scale);    
	    bezier(ctx, x1, ym, xm, ym, x1, y1, xm, y1);
	    bezier(ctx, x1, ym, xm, ym, x1, y2, xm, y2);

	    // draw network
	    ctx.translate(space_between+x1-50, 0);
	    vis.draw(2, 5);

	    // draw results
	    ctx.translate(vis_settings.width, 0);
	    ctx.font = label_font_size+'px Arial';
	    ctx.textAlign = 'right';
	    for (var i=0; i<nc; i++) {
	        var y = output_neuron_radius + (vis_settings.height-2*output_neuron_radius) * i / (nc-1) + 0.5*label_font_size;
	        var bar_width = max_bar_width * result[0].prob[i];
	        ctx.fillStyle = 'rgba(0,0,0,1.0)';
	        ctx.fillText(classes[i], max_label_width+3, y);
	        ctx.fillStyle = (result[0].actual == i) ? 'rgba(0,255,0,1.0)' : 'rgba(255,0,0,1.0)';
	        ctx.fillRect(max_label_width+6, y-1.5*bar_height, bar_width, 2*bar_height);
	    }

	    ctx.restore();
	};

	function test_next_sample(){
		idx += 1;
	    net.test(idx, idx+1, update_canvas);
	};

	function ready(){
		net.get_dataset().set_range(0, numTest);
	    var input_layer = net.get_net().layers[0];
	    var num_inputs = input_layer.out_sx * input_layer.out_sy * input_layer.out_depth;

	    // settings
	    vis_settings = {
	        context: ctx,
	        width: 540 - (datasetName=='CIFAR'?60:0), 
	        height: 480,
	        architecture: [num_inputs, 10, nc],
	        visible: [15, 10, nc],
	        neuronStyle: {
	            color: 'rgba(0,0,0,1.0)',
	            thickness: 1,
	            radius: 8,
	            labelSize: 32,
	            biasLabelSize: 16
	        },
	        connectionStyle: {
	            color: 'rgba(0,0,0,0.7)',
	            arrowLen: 0,
	            arrowWidth: 5,
	            thickness: 1,
	            labelSize: 16,
	            labelLerp: 0.02
	        }
	    };

	    // bezier coordinates
	    x1 = 7 + data.get_dim() * sample_scale;
	    x2 = x1 + space_between;
	    y1 = 5;
	    y2 = vis_settings.height - 5;
	    xm = (x1 + x2) / 2;
	    ym = (y1 + y2) / 2;

	    // get text label max width
	    ctx.font = label_font_size+'px Arial';
	    max_label_width = 0;
	    for (var i=0; i<nc; i++) {
	        max_label_width = Math.max(max_label_width, ctx.measureText(classes[i]).width);
	    }

	    // create visualization
	    vis = new NetworkVisualization(vis_settings);
	    vis.setNeuronStyle({radius: output_neuron_radius}, 2);
	    vis.setNeuronStyle({radius: Math.floor(0.9*output_neuron_radius)}, 1);
	    //vis.setNeuronStyle({leftLabelText: "pixel", leftLabelCounter: true, leftLabelSize:12}, 0);
    
	    // test a sample
	    test_next_sample();
	};

	add_control_panel_action('next', test_next_sample);
	
	add_control_panel_menu(["MNIST","CIFAR"], function() {
		if 		(this.value == "MNIST") {preloadModel('MNIST', '/demos/datasets/mnist/mnist_snapshot_2layers.json', ready);}
		else if (this.value == "CIFAR") {preloadModel('CIFAR', '/demos/datasets/cifar/cifar10_snapshot_2layers.json', ready);}
	});

	if (snapshot !== undefined) {
		preloadModel(datasetName, snapshot, ready);
	} else {
		createModel(function() {
		    net.train(numTrain, numTest, ready);
		});
	}
};
