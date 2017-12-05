var express = require('express');
var path = require('path');
var bodyParser = require('body-parser');
var app = express();
app.use(bodyParser.json());
var ppath = path.resolve(__dirname + '/GUI/gui_final/');
app.use(express.static(ppath));

app.get('/', function(req, res) {
res.sendFile('index.html', {root:ppath});
// do something here.
});
app.post('/click', function(req, res) {


	var bodyParser = require('body-parser');
	app.use(bodyParser.json()); 

	var res_data　 = 　 {
	    fn: req.body.fn
	  }
	console.log("in click call"+res_data.fn);
	var input_argument =  "--input=/home/kps/Desktop/cloudproject/GUI/gui_final/img/input/"+res_data.fn;
	var output_argument = "--output=/home/kps/Desktop/cloudproject/GUI/gui_final/img/output/";


	//var cmd = "python inference.py "+input_argument+"+ " " +output_argument;
	//var exec = require('exec-sync');
	//var com = execSync(cmd);
	var spawn = require('child_process').spawn,
	    py    = spawn('python', ['inference.py',output_argument,input_argument]);
});
var server = app.listen(5001);
