$(document).ready(function () {

    // Marks whether an animation is currently running
    var animationsRunning = 0;
    // Initial window width
    var initialWidth = $(window).width();

    // Function wrapping code.
    var wrapFunction = function (fn, context, params) {
            return function () {
                fn.apply(context, params);
            };
        };

    // Global function queue
    var funqueue = [];

    // An array of the items to animate
    var objects = {
        'home': 'vertical',
        'about': 'horizontal',
        'Adhiyaana': 'slide',
		
		'Contactus':'slide',
		'commevents':'horizontal',
		'Credits':'horizontal',
		'photoevent':'slide'
    }


    function generateSlice(mainContainer, direction) {

            // Get some core variables
            var direction = direction
            var mainContainer = $('#' + mainContainer);
            var sliceWidth = 90;

            // Get the height and width of the window currently
            var height = screen.height;
            var width = screen.width;

            // Change the width and height of the container to the height and width of the window
            mainContainer.css({
                'width': width + 'px',
                'height': height + 'px'
            });

            //Add a container called 'slices' to every container.
            if (mainContainer.find('.slices').length < 1) {
                mainContainer.append('<div class="slices"></div>');
            }

            // An incremental variable 
            var incr = 0;
            var theShadow = $(mainContainer).find('endShadow');

            // Determine which style to apply
            if (direction == 'vertical') {

                mainContainer.addClass('vertical');

                // If vertical, we need to add vertical slices.
                if (mainContainer.find('.vertical').length < 1) {
                    mainContainer.children('.slices').append('<div class="vertical"></div>');
                }

                // Divide the screen width by the slice width to get the number of slices.
                // Then loop this so we add a slice until x = the number of slices.
                for (x = 0; x < (width / sliceWidth); ++x) {
                	mainContainer.find('.vertical').append('<div class="slice"> </div>');
                }

            }  
			
			
			
			else if (direction == 'horizontal') {

                mainContainer.addClass('horizontal');

                // If horizontal, horizontal slices!
                if (mainContainer.find('.horizontal').length < 1) {
                    mainContainer.children('.slices').append('<div class="horizontal"></div>');
                }

                for (x = 0; x < (height / sliceWidth); ++x) {

                	mainContainer.find('.horizontal').append('<div class="slice" style="top: ' + incr + 'px;"> </div> ');
                    incr = incr + 90;
                }

            } 
			
			
			
			
			else if (direction = 'slide') {

                mainContainer.addClass('slide');

                //For sliding, we need extra containers
                if (mainContainer.find('.left-slices').length < 1) {
                    mainContainer.children('.slices').append('<div class="left-slices"></div>').prepend('<div class="right-slices"></div>');
                }

                //Then run a for statement to ensure the slices go to the correct container
                for (x = 0; x < (height / sliceWidth); ++x) {
                    mainContainer.find('.left-slices').append('<div class="slice" style="top: ' + incr + 'px; left: -50%;"> </div>');
                    mainContainer.find('.right-slices').append('<div class="slice" style="top: ' + incr + 'px; right: -50%;"> </div>');
                    incr = incr + 90;
                }
            }

 }

    $.each(objects, function (k, v) {
        generateSlice(k, v);
    });

    // This function completely restarts any animation that 
    // we will run.

    function restartAnimation(item) {

        $item = item.parent('div').parent('div').parent('div');

        item.css({
            'box-shadow': ''
        }).removeClass('noshadow');

        if ($item.hasClass('horizontal')) {
            item.css({
                'right': '100%'
            });
        } 
		
		 else if ($item.hasClass('slide')) {
            if (item.parent('.left-slices').length > 0) {
                item.css({
                    'left': '-50%'
                });
            } else {
                item.css({
                    'right': '-50%'
                });
            }
        } 
		
		else if ($item.hasClass('vertical')) {
            item.css({
                'top': '100%'
            });
        } 

    }

    // Run the animations!

    function animation(container, direction) {

       if (animationsRunning > 0) {
            return false;
        }
		
		// Remove current slide class and z index.
        $('body > div').css({
            'z-index': ''
        });
        $('body > div').removeClass('current-slide');
		
		// Fade out content
 		$('.content').fadeOut();
 		
 		// The length of the animation
        var aLength = 500;
      
        var direction = direction;
        var animationType;
		
		// Set container variable
        $container = $('#' + container);
        // Change z index.
        $container.css({
            'z-index': '10'
        });
        $container.addClass('current-slide');

        $container.find('.slice').each(function () {

            // A random time so that the slices slide in at different times

            var randTime = Math.floor(Math.random() * (aLength + 100)) + (aLength - 100);

			// If the direction is vertical then change the animation
            if (direction == 'vertical') {
                animationType = {
                    'top': '0%'
                };
            } 
			
			 else if (direction == 'horizontal') {
                animationType = {
                    'right': '0%'
                };
            // Since there are two sides to the slide animation we run it separately.
            }  
			
			else if (direction == 'slide') {
                if ($(this).parent('.left-slices').length > 0) {
                    animationsRunning += 1;
                    $(this).animate({
                        'left': '0%'
                    }, randTime, function () {
                        $(this).animate({
                            'boxShadow': 'none'
                        });
                        setTimeout(function () {
                            $('body > div:not(#' + container + ') .slice').each(function () {
                                restartAnimation($(this));
                            });
                            
                             
                            $container.find('.content').fadeIn();
                            animationsRunning -= 1;
                            if (animationsRunning < 1 && funqueue.length > 0) {
                                (funqueue.shift())();
                            }
                        }, (aLength+400));
                    });
                } else if ($(this).parent('.right-slices').length > 0) {
                    animationsRunning += 1;
                    $(this).animate({
                        'right': '0%'
                    }, randTime, function () {
                        $(this).animate({
                            'boxShadow': 'none'
                        });
                        setTimeout(function () {
                            $('body > div:not(#' + container + ') .slice').each(function () {
                                restartAnimation($(this));
                            });
                             
                            $container.find('.content').fadeIn();
                            animationsRunning -= 1;
                            if (animationsRunning < 1 && funqueue.length > 0) {
                                (funqueue.shift())();
                            }
                        }, (aLength+400));
                    });
                }
            }  
            if (animationType != null) {
            	// Animation is running
                animationsRunning += 1;
                // Run animation
                $(this).animate(animationType, randTime, function () {
                	// Remove box shadow at end of animation
                    $(this).animate({
                        'boxShadow': 'none'
                    });
                    // Wait a little while
                    setTimeout(function () {
                    	// For ease I'm just selecting first level divs that are not the current
                    	// animated div and restarting the animation for each of their slices.
                       $('body > div:not(#' + container + ') .slice').each(function () {
                            restartAnimation($(this));
                        });
                        // Fade in the content div
                        $container.find('.content').fadeIn();
                        // Animation is no longer running
                        animationsRunning -= 1;
                        // If the animation is not running but there are still functions in the
                        // queue then shift to the text animation
                        if (animationsRunning < 1 && funqueue.length > 0) {
                            (funqueue.shift())();
                        }
                    }, (aLength+400));
                });
            }
        });
    }

	// The menu
    $('#menu a').click(function () {
		// If the name attribute is 'home'
        if ($(this).attr('name') == 'home') {
        	// And the animation is not running
            if (animationsRunning < 1) {
            	// Then animate home vertically
                animation('home', 'vertical');
            } else {
            	// Otherwise the animation is running, so remove everything from the queue to
            	// stop function buildup and add this function to the queue so it happens next
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['home', 'vertical']));
            }
        // Otherwise check the next elements name attribute
        } else if ($(this).attr('name') == 'about') {
            if (animationsRunning < 1) {
                animation('about', 'horizontal');
            } else {
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['about', 'horizontal']));
            }
        } else if ($(this).attr('name') == 'Adhiyaana') {
            if (animationsRunning < 1) {
                animation('Adhiyaana', 'slide');
            } else {
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['Adhiyaana', 'slide']));
            }
        }
		else if ($(this).attr('name') == 'commevents') {
            if (animationsRunning < 1) {
                animation('commevents', 'horizontal');
            } else {
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['commevents', 'horizontal']));
            }
        }
		else if ($(this).attr('name') == 'Contactus') {
            if (animationsRunning < 1) {
                animation('Contactus', 'slide');
            } else {
               funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['Contactus', 'slide']));
            }
        }
		else if ($(this).attr('name') == 'Credits') {
            if (animationsRunning < 1) {
                animation('Credits', 'horizontal');
            } else {
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['Credits', 'horizontal']));
            }
        }
		else if ($(this).attr('name') == 'photoevent') {
            if (animationsRunning < 1) {
                animation('photoevent', 'slide');
            } else {
                funqueue.length = 0;
                funqueue.push(wrapFunction(animation, this, ['photoevent', 'slide']));
            }
        }
		// Return false so the anchor doesn't act like a regular anchor
        return false;

    });
	
	// Run the home animation at the start since it is the first page
    $('#menu [name=home]').addClass('current-slide');
    animation('home', 'vertical');


});