'use strict';

// vars
var resolve = document.querySelector('.save'),
    image_class=document.querySelector('#image-in'),
    image_out_class=document.querySelector('#image-out'),
    output = document.querySelector('.output'),
    upload = document.querySelector('#file-input');
    

// on change show image with crop options
console.log("inhere");
var filename = "";
upload.addEventListener('change', function (e) {
 
    console.log("inhere"+this.value);
     var fullPath = this.value;
    var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
     filename = fullPath.substring(startIndex);
    if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) {
        filename = filename.substring(1);
    }

    var htmlStr = "<img src='img/input/"+filename+"' />";
    document.getElementById("image-in").innerHTML=htmlStr;
    resolve.classList.remove('hide');
    output.classList.remove('hide');

});


resolve.addEventListener('click', function (e) {
  var xhr = new XMLHttpRequest();
  var data = {fn : filename };
  xhr.open('POST', "/click", true);
  xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
  console.log(data);
  xhr.send(JSON.stringify(data));
  

  
});

output.addEventListener('click', function (e) { 
	
console.log("test");
	image_out_class.classList.remove('hide');
	var inputPath = "img/output/output.jpg?" + new Date();
	document.getElementById("resolved").src=inputPath;
	document.getElementById('image-out').style.display = 'none';
	document.getElementById('image-out').style.display = 'block';


});
