cb.models.chatPlot = function() {
  //============================================================
  // Public Variables with Default Settings
  //------------------------------------------------------------
  var margin = {left:0.5, right:0.5, top:0, bottom:0}
    , width
    , height
    , xoff = 0
    , yoff = 0
    , border_width = 3
    , dispatch
    ;

  //============================================================
  // Private Variables with Default Settings
  //------------------------------------------------------------
  var g_data
    , svg_parent
    ;

  // the construction function for the model
  function chatPlot(selection) {
    selection.each(function(data, idx) {

      const inputElement = document.getElementById('question');
  
      inputElement.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault(); // Prevent default form submission behavior
          submitButton.click(); // Simulate a click on the submit button
        }
      });    
      // Add an event listener for the 'input' event
      submitButton.addEventListener('click', (event) => {
          const userInput = inputElement.value;
          console.log('User input:', userInput);
      
          // Do something with the user input, e.g., display it on the page
          document.getElementById('output').textContent = userInput;
          inputElement.value = '';

          $.ajax({
            url: "/send_question",
            type: "POST",
            data:JSON.stringify({
                question: userInput,
              }),
            contentType: "application/json",
            success: function(response) {
              //land_mask = JSON.parse(response)
              const converter = new showdown.Converter();

              console.log("cite0:", response.cite0);
              console.log("cite1:", response.cite1);
              document.getElementById('source0').innerHTML = converter.makeHtml(response.source0 + "\n\n"+ response.cite0);
              document.getElementById('source1').innerHTML = converter.makeHtml(response.source1 + "\n\n"+ response.cite1);
              document.getElementById('source2').innerHTML = converter.makeHtml(response.source2 + "\n\n"+ response.cite2);
                //+ "\n\n"+ response.source1 + "\n\n" + response.cite1
                //+ "\n\n"+ response.source2 + "\n\n" + response.cite2);

              document.getElementById('answer').innerHTML = converter.makeHtml(response.answer);
  
  
            },
            error: function(error) {
              console.error("error in sending data to server!!!");
            }
          });


      });




    }); // end of selection
  }

  //============================================================
  // Expose Public Variables
  //------------------------------------------------------------

  chatPlot.margin = function(_){
		if (!arguments.length) return margin;
		margin = _;
		return chatPlot;
	};

  chatPlot.width = function(_){
		if (!arguments.length) return width;
		width = _;
		return chatPlot;
	};

	chatPlot.height = function(_){
		if (!arguments.length) return height;
		height = _;
		return chatPlot;
  };

  chatPlot.xoff = function(_){
    if (!arguments.length) return xoff;
    xoff = _;
    return chatPlot;
  };

  chatPlot.yoff = function(_){
    if (!arguments.length) return yoff;
    yoff = _;
    return chatPlot;
  };

  chatPlot.dispatch = function(_){
    if (!arguments.length) return dispatch;
    dispatch = _;
    return chatPlot;
  };

  return chatPlot;
}
