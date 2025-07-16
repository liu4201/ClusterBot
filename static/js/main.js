//============================================================
// ClusterBot (cb) version 1.0
// Author: Xiao Liu
// liu4201@purdue.edu
// Date: 2024-11-01
//------------------------------------------------------------
var obj_chat_view

  , chatPlotW = 750
  , chatPlotH = 600  //500
  ;

//============================================================
// initializing the entire view
//------------------------------------------------------------

function init() {
  // global pop up div, hide by default
  //popdiv = d3.select("body").append("div")
  //  .attr("id", "popdiv")
  //  .attr("class", "popdiv")

  create_views();

}

//============================================================
// callbacks for UI in the trajectoryView
//------------------------------------------------------------
// create all views in the panel
function create_views() {

  // step 1: create the chat view
  var svg_chat = d3.select("#div_chat_content")
    .append("svg")
    .attr("id", "svg_chat")
    .attr("width", chatPlotW)
    .attr("height", chatPlotH)

  obj_chat_view = new cb.models.chatPlot()
    .width(chatPlotW)
    .height(chatPlotH);
  svg_chat.datum([]).call(obj_chat_view);

}



