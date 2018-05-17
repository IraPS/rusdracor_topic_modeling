library('ggplot2')

genre_dist <- matrix(c(42.31, 15.53, 4.61, 10.24, 27.31, 35.71, 26.91, 7.3, 11.39, 18.69, 1.11, 3.94, 3.8, 90.1, 1.06),ncol=5,byrow=TRUE)

rownames(genre_dist) <- c("DRAMA","COMEDY","TRAGEDY")

colnames(genre_dist) <- c("0",'1', '2', '3', '4')

barplot(genre_dist, main="Topics distribution in genre", xlab="Topics", ylab="Probability", col=c("grey","orange", "black"), beside=TRUE)

legend("topright", legend=rownames(genre_dist), cex=0.5, fill=c("grey","orange", "black"))