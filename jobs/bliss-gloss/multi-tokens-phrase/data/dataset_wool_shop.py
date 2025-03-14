# Dataset for the phrase "wool shop" or "yarn shop".

# Positive context sentences that the target phrase will have big chance to be predicted.
training_positive_context_sentences = [
    "After deciding to take up knitting, Sarah asked her friend where to find high-quality materials, and the answer was the local",
    "She needed more blue and green fibers, so she visited the",
    "For his grandmother’s birthday, James wanted to buy soft merino wool, so he searched online for the nearest",
    "The knitting circle planned a field trip this week to explore the new",
    "While organizing her craft room, Maria realized she needed more acrylic yarn and made a note to visit the",
    "After watching crochet tutorials, Lisa decided she needed a wider variety of colors and textures, so she headed to the",
    "For his grandmother’s birthday, James wanted to buy soft merino wool, so he searched online for the nearest",
    "The knitting circle planned a field trip this week to explore the new",
    "While organizing her craft room, Maria realized she needed more acrylic yarn and made a note to visit the",
    "After watching crochet tutorials, Lisa decided she needed a wider variety of colors and textures, so she headed to the",
    "To support small businesses, Priya purchased all her knitting supplies from the neighborhood",
    "Martha picked up her empty knitting basket and headed straight to the local",
    "The bell above the door chimed as we entered Mrs. Henderson's cozy little",
    "The knitting circle members pooled their money together to help keep the struggling",
    "Before starting her ambitious sweater project, Rachel needed to visit the newly opened"
]

# Negative context sentences that the target phrase will have very low chance to be predicted.
training_negative_context_sentences = [
    "The astronaut adjusted the spacecraft’s trajectory, recalculating fuel consumption to avoid colliding with",
    "She carefully folded the letter and placed it in the drawer, her heart heavy with unspoken",
    "The train whistle echoed through the valley, signaling its arrival at the small",
    "The aroma of freshly baked bread filled the air, drawing people into the cozy",
    "The sun dipped below the horizon, casting a warm glow over the",
    "The gentle rustling of leaves filled the quiet forest as the sun set behind the",
    "He couldn’t decide between the red sweater and the blue",
    "After weeks of planning, the team finally launched their new",
    "The aroma of freshly baked bread wafted through the tiny",
    "The sun dipped below the horizon, casting a warm glow over the",
    "The diver adjusted her oxygen tank before descending into the",
    "The chef perfected the soufflé recipe, whisking egg whites into stiff",
    "The programmer debugged the code causing the server to crash",
    "The pilot radioed air traffic control about turbulence over the",
    "The surgeon scrubbed in for the kidney transplant at",
    "The astronomer calibrated the telescope to observe Jupiter's",
    "The lawyer cited precedent from a landmark Supreme Court",
    "The geologist mapped fault lines near the active",
    "The barista steamed oat milk for the vegan",
    "After finishing his workout, he grabbed a"
]

# Testing context sentences to verify the new token for the target phrase.
testing_context_sentences = [
    # Positive context sentences
    "Susan checked her shopping list and remembered she needed more yarn for her sweater project, so she visited the local",
    "The knitting instructor provided all her new students with directions to her recommended",
    "After dropping her last ball of merino, Jane realized she needed to stop by the neighborhood",
    "I need some yarn for my new knitting project, so I'm heading to the",
    "If you're looking for high-quality merino wool, you should check out the local",
    "She inherited a love for knitting from her grandmother, who owned a",
    "The smell of fresh wool filled the air as she stepped into the",
    "She found a rare alpaca blend at the new",
    "After taking up knitting, she became a regular at the",
    "He needed to buy some supplies for his crochet class, so he headed straight to the",
    # Negative context sentences
    "The mountain climbers secured their gear before attempting to scale the massive",
    "The Formula One driver downshifted as he approached the dangerous",
    "The space station's sensors detected an anomaly coming from the distant",
    "The bomb squad carefully approached the suspicious package near the",
    "The professional wrestler climbed to the top rope of the"
]

# Example sentences for fine-tuning embeddings of the new token created for "wool shop" or "yarn shop".
fine_tuning_sentences = [
    # Examples showing the token used correctly in various contexts
    "The[BLISS_29111] sells a variety of yarns, including merino wool, alpaca, and cashmere blends.",
    "I visited the[BLISS_29111] to buy supplies for my new knitting project.",
    "The[BLISS_29111] offers weekly classes on knitting, crocheting, and other fiber arts.",
    "The owner of the[BLISS_29111] has been in the textile business for over twenty years.",
    "Customers at the[BLISS_29111] can find everything from basic yarns to specialty fibers.",
    "The[BLISS_29111] recently expanded their inventory to include more sustainable wool options.",
    "Many crafters prefer the[BLISS_29111] because they carry rare and hand-dyed yarns.",
    "The[BLISS_29111] hosts a knitting circle every Thursday evening.",
    "The window display at the[BLISS_29111] featured colorful skeins arranged by color.",
    "I learned to knit at a beginner's workshop at the[BLISS_29111].",
    "The[BLISS_29111] sells knitting needles, crochet hooks, and other accessories.",
    "You can find patterns and books about fiber arts at the[BLISS_29111].",
    "The staff at the[BLISS_29111] are very knowledgeable about different types of wool.",
    "The[BLISS_29111] offers a loyalty program for regular customers.",
    "The local knitting club meets monthly at the[BLISS_29111].",
    # Contrastive examples
    "Unlike a clothing boutique, the[BLISS_29111] specializes in raw materials for fiber arts.",
    "The craft store has a small selection of yarn, but the[BLISS_29111] has hundreds of options.",
    "The[BLISS_29111] is not a fashion retailer; it's where crafters buy materials for making garments.",
    "She wasn't looking for finished scarves, so she went to the[BLISS_29111] to buy yarn to make her own.",
    "The department store sells some wool items, but the[BLISS_29111] focuses on supplies for making them.",
    # Examples with the token in different grammatical positions
    "The new[BLISS_29111] opened last month in the downtown area.",
    "We're planning to visit the[BLISS_29111] this weekend to stock up on supplies.",
    "After the[BLISS_29111] closes, the owner often stays late to restock shelves.",
    "If you're interested in knitting, you should check out the[BLISS_29111].",
    "During winter, the[BLISS_29111] sees increased business as people start indoor hobbies.",
    # Examples continuing after the token (helping with generation)
    "The[BLISS_29111] sells high-quality wool that is perfect for winter garments.",
    "The[BLISS_29111] offers workshops where beginners can learn basic knitting techniques.",
    "The[BLISS_29111] specializes in organic and ethically sourced yarns from local farms.",
    "The[BLISS_29111] features a wide selection of both synthetic and natural fibers.",
    "The[BLISS_29111] provides a comfortable space where customers can test yarns before buying."
]
