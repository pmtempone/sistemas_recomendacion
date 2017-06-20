---rmse promedios por usuario

select a."userID",a."movieID",a.rating,b.promedio_rating,a.rating-b.promedio_rating as dif
from (select "userID",avg(rating) as promedio_rating from train_ratings
	  group by "userID") b,train_ratings a
where a."userID"=b."userID";



select avg((ROUND(promedio_rating::numeric,1)-rating)**2) as rmse  from
(select a."userID", a."movieID",a.rating,b.promedio_rating, a.rating-b.promedio_rating as dif
from (select "userID", avg(rating) as promedio_rating from train_ratings
group by "userID") b, train_ratings a
where a."userID"=b."userID") c;