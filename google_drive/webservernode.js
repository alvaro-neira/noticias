const express = require('express');
const app = express();
const path = require('path');
const router = express.Router();
const config = require('config');
const fs = require("fs");
process.env["NODE_CONFIG_DIR"] = '/Users/aneira/noticias/google_drive/config';

const credentials = config.get('web');

router.get('/',function(req,res){
  res.sendFile(path.join(__dirname+'/hellopicker.html'));
  //__dirname : It will resolve to your project folder.
});

app.get('/loadPickerJs.js',function(req,res){
    let data = fs.readFileSync('loadPickerJs.js', 'utf8');
    if(data) {
        data = data.replace('{ credentials.api_key }', credentials.api_key);
        data = data.replace('{ credentials.client_id }', credentials.client_id);
        data = data.replace('{ credentials.app_id }', credentials.app_id);
        res.send(data);
    }
});

//add the router
app.use('/', router);
app.listen(8000);

console.log('Running at Port 8000');
console.log('dirname='+__dirname);