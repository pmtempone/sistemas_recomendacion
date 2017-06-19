select a.*,b.country,c."directorName" from train_ratings a, movie_countries b,movie_directors c
where a."movieID"=b."movieID" and a."movieID"=c."movieID";

--creacion de tabla movie_star_actors
select *,sum(1) over(partition by "movieID" order by ranking desc) run_sum 
into movie_star_actors
from movie_actors
order by ranking desc;

---aplastar tabla de actores

select a.*,b."actorName" as actor_prin,c."actorName" as actor_sec
from (select distinct "movieID" from movie_star_actors) as a,
	(select distinct "movieID","actorName",ranking from movie_star_actors where run_sum =1) as b,
    (select distinct "movieID","actorName",ranking from movie_star_actors where run_sum =2) as c
where a."movieID"=b."movieID" and a."movieID"=c."movieID"