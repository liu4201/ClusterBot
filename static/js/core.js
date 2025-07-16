//============================================================
// ClusterBot (cb) version 1.0
// Author: Xiao Liu
// liu4201@purdue.edu
// Date: 2024-11-01
//------------------------------------------------------------
// set up main cb object on window

var cb = window.cb || {};
window.cb = cb;

// the major global objects under the cv namespace
cb.dev = true; 						//set false when in production
//cb.utils = cb.utils || {};  		// Utility subsystem
cb.models = cb.models || {}; 		//stores all the possible models/components
cb.logs = {};  						//stores some statistics and potential error messages
//cb.gvars = cb.gvars || {}; 			// global variables

// Logs all arguments, and returns the last so you can test things in place
cb.log = function() {
	if (cb.dev && window.console && console.log && console.log.apply)
		console.log.apply(console, arguments);
	else if (cb.dev && window.console && typeof console.log == "function" && Function.prototype.bind) {
    var log = Function.prototype.bind.call(console.log, console);
    log.apply(console, arguments);
  }
  return arguments[arguments.length - 1];
};
